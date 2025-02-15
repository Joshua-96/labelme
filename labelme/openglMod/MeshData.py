from qtpy import QtGui
from time import perf_counter
import numpy as np
try:
    import cupy as cp
    mode = "cp"
except ImportError:
    mode = "np"


class MeshData(object):
    """
    Class for storing and operating on 3D mesh data. May contain:

      - list of vertex locations
      - list of edges
      - list of triangles
      - colors per vertex, edge, or tri
      - normals per vertex or tri

    This class handles conversion between the standard [list of vertexes,
    list of faces] format (suitable for use with glDrawElements) and 'indexed'
    [list of vertexes] format (suitable for use with glDrawArrays).
    It will automatically compute face normal vectors as well as averaged
    vertex normal vectors.

    The class attempts to be as efficient as possible in caching conversion
    results and avoiding unnecessary conversions.
    """

    def __init__(self, vertexes=None, faces=None, edges=None,
                 vertexColors=None, faceColors=None, meshWidth=None):
        """
        ==============  =====================================================
        **Arguments:**
        vertexes        (Nv, 3) array of vertex coordinates.
                        If faces is not specified, then this will instead be
                        interpreted as (Nf, 3, 3) array of coordinates.
        faces           (Nf, 3) array of indexes into the vertex array.
        edges           [not available yet]
        vertexColors    (Nv, 4) array of vertex colors.
                        If faces is not specified, then this will instead be
                        interpreted as (Nf, 3, 4) array of colors.
        faceColors      (Nf, 4) array of face colors.
        ==============  =====================================================

        All arguments are optional.
        """
        self._vertexes = None  # (Nv,3) array of vertex coordinates
        # (Nf, 3, 3) array of vertex coordinates
        self._vertexesIndexedByFaces = None
        # (Ne, 2, 3) array of vertex coordinates
        self._vertexesIndexedByEdges = None

        # mappings between vertexes, faces, and edges
        # Nx3 array of indexes into self._vertexes specifying three vertexes
        # for each face
        self._faces = None
        self._flattendFaces = None
        # Nx2 array of indexes into self._vertexes specifying two vertexes
        # per edge
        self._edges = None
        # maps vertex ID to a list of face IDs (inverse mapping of _faces)
        self._vertexFaces = None
        # maps vertex ID to a list of edge IDs (inverse mapping of _edges)
        self._vertexEdges = None

        # Per-vertex data
        # (Nv, 3) array of normals, one per vertex
        self._vertexNormals = None
        self._vertexNormalsIndexedByFaces = None  # (Nf, 3, 3) array of normals
        self._vertexColors = None                 # (Nv, 3) array of colors
        self._vertexColorsIndexedByFaces = None   # (Nf, 3, 4) array of colors
        self._vertexColorsIndexedByEdges = None   # (Nf, 2, 4) array of colors

        # Per-face data
        self._faceNormals = None                # (Nf, 3) array of face normals
        # (Nf, 3, 3) array of face normals
        self._faceNormalsIndexedByFaces = None
        self._faceColors = None                 # (Nf, 4) array of face colors
        # (Nf, 3, 4) array of face colors
        self._faceColorsIndexedByFaces = None
        # (Ne, 2, 4) array of face colors
        self._faceColorsIndexedByEdges = None

        # Per-edge data
        self._edgeColors = None                # (Ne, 4) array of edge colors
        # (Ne, 2, 4) array of edge colors
        self._edgeColorsIndexedByEdges = None
        # self._meshColor = (1, 1, 1, 0.1)  # default color to use if no
        # face/edge/vertex colors are given

        self.meshWidth = meshWidth

        if vertexes is not None:
            if faces is None:
                self.setVertexes(vertexes, indexed='faces')
                if vertexColors is not None:
                    self.setVertexColors(vertexColors, indexed='faces')
                if faceColors is not None:
                    self.setFaceColors(faceColors, indexed='faces')
            else:
                self.setVertexes(vertexes)
                self.setFaces(faces)
                if vertexColors is not None:
                    self.setVertexColors(vertexColors)
                if faceColors is not None:
                    self.setFaceColors(faceColors)

    def faces(self):
        """Return an array (Nf, 3) of vertex indexes, three per triangular
           face in the mesh.

        If faces have not been computed for this mesh, the function
        returns None.
        """
        return self._faces

    def flattend_faces(self):
        return self._flattendFaces

    def edges(self):
        """Return an array (Nf, 3) of vertex indexes, two per edge
        in the mesh.
        """
        if self._edges is None:
            self._computeEdges()
        return self._edges

    def setFaces(self, faces):
        """Set the (Nf, 3) array of faces. Each rown in the array contains
        three indexes into the vertex array, specifying the three corners
        of a triangular face."""
        self._faces = faces
        self._flattendFaces = faces.flatten()
        self._edges = None
        self._vertexFaces = None
        self._vertexesIndexedByFaces = None
        self.resetNormals()
        self._vertexColorsIndexedByFaces = None
        self._faceColorsIndexedByFaces = None

    def vertexes(self, indexed=None):
        """Return an array (N,3) of the positions of vertexes in the mesh.
        By default, each unique vertex appears only once in the array.
        If indexed is 'faces', then the array will instead contain three
        vertexes per face in the mesh (and a single vertex may appear
        more than once in the array)."""
        if indexed is None:
            if self._vertexes is None and \
                    self._vertexesIndexedByFaces is not None:
                self._computeUnindexedVertexes()
            return self._vertexes
        elif indexed == 'faces':
            if self._vertexesIndexedByFaces is None and \
                    self._vertexes is not None:
                self._vertexesIndexedByFaces = self._vertexes[self.faces()]
            return self._vertexesIndexedByFaces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def setVertexes(self, verts=None, indexed=None, resetNormals=True):
        """
        Set the array (Nv, 3) of vertex coordinates.
        If indexed=='faces', then the data must have shape (Nf, 3, 3) and is
        assumed to be already indexed as a list of faces.
        This will cause any pre-existing normal vectors to be cleared
        unless resetNormals=False.
        """
        if indexed is None:
            if verts is not None:
                self._vertexes = np.ascontiguousarray(verts, dtype=np.float32)
            self._vertexesIndexedByFaces = None
        elif indexed == 'faces':
            self._vertexes = None
            if verts is not None:
                self._vertexesIndexedByFaces = np.ascontiguousarray(
                    verts, dtype=np.float32)
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

        if resetNormals:
            self.resetNormals()

    def resetNormals(self):
        self._vertexNormals = None
        self._vertexNormalsIndexedByFaces = None
        self._faceNormals = None
        self._faceNormalsIndexedByFaces = None

    def hasFaceIndexedData(self):
        """Return True if this object already has vertex positions
        indexed by face"""
        return self._vertexesIndexedByFaces is not None

    def hasEdgeIndexedData(self):
        return self._vertexesIndexedByEdges is not None

    def hasVertexColor(self):
        """Return True if this data set has vertex color information"""
        for v in (self._vertexColors, self._vertexColorsIndexedByFaces,
                  self._vertexColorsIndexedByEdges):
            if v is not None:
                return True
        return False

    def hasFaceColor(self):
        """Return True if this data set has face color information"""
        for v in (self._faceColors, self._faceColorsIndexedByFaces,
                  self._faceColorsIndexedByEdges):
            if v is not None:
                return True
        return False

    def faceNormals(self, indexed=None, mode="np"):
        """
        Return an array (Nf, 3) of normal vectors for each face.
        If indexed='faces', then instead return an indexed array
        (Nf, 3, 3)  (this is just the same array with each vector
        copied three times).
        """

        if self._faceNormals is None:
            if mode == "np":
                v = np.array(self.vertexes(indexed='faces').astype(np.float32))
                self._faceNormals = np.cross(
                    v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            elif mode == "cp":
                v = cp.array(self.vertexes(indexed='faces').astype(np.float32))
                self._faceNormals = cp.cross(
                    v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])

        if indexed is None:
            return self._faceNormals
        elif indexed == 'faces':
            if self._faceNormalsIndexedByFaces is None:
                norms = np.empty(
                    (self._faceNormals.shape[0], 3, 3), dtype=np.float32)
                norms[:] = self._faceNormals[:, np.newaxis, :]
                self._faceNormalsIndexedByFaces = norms

            return self._faceNormalsIndexedByFaces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def vertexNormals(self, indexed=None):
        """
        Return an array of normal vectors.
        By default, the array will be (N, 3) with one entry per unique vertex
        in the mesh.If indexed is 'faces', then the array will contain three
        normal vectors per face (and some vertexes may be repeated).
        """

        if self._vertexNormals is None:
            t0 = perf_counter()
            # faceNorms = numlib.array(self.faceNormals())
            faceNorms = self.faceNormals(mode=mode)
            t1 = perf_counter()
            # vertFaces = self.vertexFaces()
            # norm_tuple = self.get_normTuple(faceNorms)
            if mode == "cp":
                norm_tuple = cp.array(self.get_normTuple(faceNorms, mode=mode))
                t2 = perf_counter()
                sumed_norms = cp.sum(norm_tuple, axis=1)
                self._vertexNormals = sumed_norms / cp.linalg.norm(
                    sumed_norms, axis=1)[:, np.newaxis]
            elif mode == "np":
                norm_tuple = self.get_normTuple(faceNorms)
                t2 = perf_counter()
                sumed_norms = np.sum(norm_tuple, axis=1)
                self._vertexNormals = sumed_norms / np.linalg.norm(
                    sumed_norms, axis=1)[:, np.newaxis]

            # self._vertexNormals = np.empty(
            #     self._vertexes.shape,
            #     dtype=np.float32)
            t3 = perf_counter()
            # previous code from libray
            # for vindex in range(self._vertexes.shape[0]):
            #     faces = vertFaces[vindex]
            #     if len(faces) == 0:
            #         self._vertexNormals[vindex] = (0,0,0)
            #         continue
            #     norms = faceNorms[faces]  ## get all face normals
            #     norm = norms.sum(axis=0)       ## sum normals
            #     norm /= (norm**2).sum()**0.5  ## and re-normalize
            #     self._vertexNormals[vindex] = norm
            # from IPython import embed
            # embed()
            times = [t1 - t0, t2 - t1, t3 - t2]
            print(mode, times, sum(times))
        if indexed is None:
            if mode == "cp":
                return cp.asnumpy(self._vertexNormals)
            elif mode == "np":
                return self._vertexNormals
        elif indexed == 'faces':
            return self._vertexNormals[self.faces()]
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def vertexColors(self, indexed=None):
        """
        Return an array (Nv, 4) of vertex colors.
        If indexed=='faces', then instead return an indexed array
        (Nf, 3, 4).
        """
        if indexed is None:
            return self._vertexColors
        elif indexed == 'faces':
            if self._vertexColorsIndexedByFaces is None:
                self._vertexColorsIndexedByFaces = self._vertexColors[
                    self.faces()]
            return self._vertexColorsIndexedByFaces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def setVertexColors(self, colors, indexed=None):
        """
        Set the vertex color array (Nv, 4).
        If indexed=='faces', then the array will be interpreted
        as indexed and should have shape (Nf, 3, 4)
        """
        if indexed is None:
            self._vertexColors = np.ascontiguousarray(colors, dtype=np.float32)
            self._vertexColorsIndexedByFaces = None
        elif indexed == 'faces':
            self._vertexColors = None
            self._vertexColorsIndexedByFaces = np.ascontiguousarray(
                colors, dtype=np.float32)
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def faceColors(self, indexed=None):
        """
        Return an array (Nf, 4) of face colors.
        If indexed=='faces', then instead return an indexed array
        (Nf, 3, 4)  (note this is just the same array with each color
        repeated three times).
        """
        if indexed is None:
            return self._faceColors
        elif indexed == 'faces':
            if self._faceColorsIndexedByFaces is None and \
                    self._faceColors is not None:
                Nf = self._faceColors.shape[0]
                self._faceColorsIndexedByFaces = np.empty(
                    (Nf, 3, 4), dtype=self._faceColors.dtype)
                self._faceColorsIndexedByFaces[:] = self._faceColors.reshape(
                    Nf, 1, 4)
            return self._faceColorsIndexedByFaces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def setFaceColors(self, colors, indexed=None):
        """
        Set the face color array (Nf, 4).
        If indexed=='faces', then the array will be interpreted
        as indexed and should have shape (Nf, 3, 4)
        """
        if indexed is None:
            self._faceColors = np.ascontiguousarray(colors, dtype=np.float32)
            self._faceColorsIndexedByFaces = None
        elif indexed == 'faces':
            self._faceColors = None
            self._faceColorsIndexedByFaces = np.ascontiguousarray(
                colors, dtype=np.float32)
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def faceCount(self):
        """
        Return the number of faces in the mesh.
        """
        if self._faces is not None:
            return self._faces.shape[0]
        elif self._vertexesIndexedByFaces is not None:
            return self._vertexesIndexedByFaces.shape[0]

    def edgeColors(self):
        return self._edgeColors

    # def _setIndexedFaces(self, faces, vertexColors=None, faceColors=None):
        # self._vertexesIndexedByFaces = faces
        # self._vertexColorsIndexedByFaces = vertexColors
        # self._faceColorsIndexedByFaces = faceColors

    def _computeUnindexedVertexes(self):
        # Given (Nv, 3, 3) array of vertexes-indexed-by-face, convert backward
        # to unindexed vertexes This is done by collapsing into a list of
        # 'unique' vertexes (difference < 1e-14)

        # I think generally this should be discouraged..
        faces = self._vertexesIndexedByFaces
        verts = {}  # used to remember the index of each vertex position
        self._faces = np.empty(faces.shape[:2], dtype=np.uint)
        self._vertexes = []
        self._vertexFaces = []
        self._faceNormals = None
        self._vertexNormals = None
        for i in range(faces.shape[0]):
            face = faces[i]
            for j in range(face.shape[0]):
                pt = face[j]
                # quantize to be sure that nearly-identical
                # points will be merged
                pt2 = tuple([round(x * 1e14) for x in pt])
                index = verts.get(pt2, None)
                if index is None:
                    # self._vertexes.append(QtGui.QVector3D(*pt))
                    self._vertexes.append(pt)
                    self._vertexFaces.append([])
                    index = len(self._vertexes) - 1
                    verts[pt2] = index
                # keep track of which vertexes belong to which faces
                self._vertexFaces[index].append(i)
                self._faces[i, j] = index
        self._vertexes = np.array(self._vertexes, dtype=np.float32)

    # def _setUnindexedFaces(self, faces, vertexes, vertexColors=None,
    #                        faceColors=None):
    #     self._vertexes = vertexes  # [QtGui.QVector3D(*v) for v in vertexes]
    #     self._faces = faces.astype(np.uint)
    #     self._edges = None
    #     self._vertexFaces = None
    #     self._faceNormals = None
    #     self._vertexNormals = None
    #     self._vertexColors = vertexColors
    #     self._faceColors = faceColors

    def vertexFaces(self):
        """
        Return list mapping each vertex index to a list of face indexes that
        use the vertex.
        """
        if self._vertexFaces is None:
            self._vertexFaces = [[] for i in range(len(self.vertexes()))]
            for i in range(self._faces.shape[0]):
                face = self._faces[i]
                for ind in face:
                    self._vertexFaces[ind].append(i)
        return self._vertexFaces

    def get_normTuple(self, faceNorms, mode="np"):
        if mode == "np":
            numlib = np
        elif mode == "cp":
            numlib = cp
        if self._vertexFaces is None:
            # if isinstance(faceNorms, cp.ndarray):
            #     faceNorms = cp.asnumpy(faceNorms)
            normTuple = numlib.empty(
                (self._vertexes.shape[0], 6, 3),
                dtype=numlib.float32)
            normTuple.fill(numlib.nan)
            allFaces = numlib.array(self._faces)
            # vertexCount = [0 for i in range(len(self.vertexes()))]
            vertexCount = numlib.zeros(
                (len(self.vertexes())), dtype=numlib.uint8)
            step = self.meshWidth - 1
            # prepare range to have unique vertices
            rg0 = numlib.empty(
                int(self._faces.shape[0] / 2), dtype=numlib.uint32)
            rg1 = numlib.empty(
                int(self._faces.shape[0] / 2), dtype=numlib.uint32)
            for i in np.arange(0, self._faces.shape[0], 2 * step):
                i_half = int(i / 2)
                try:
                    # from IPython import embed
                    # embed()
                    rg0[i_half:i_half +
                        step] = numlib.arange(i, i + step)
                except ValueError:
                    rg0[i_half:] = numlib.arange(
                        i, self._faces.shape[0])
                    break

            for i in np.arange(step, self._faces.shape[0], 2 * step):
                i_half = int((i - step) / 2)
                try:
                    rg1[i_half: i_half +
                        step] = numlib.arange(i, i + step)
                except ValueError:
                    rg1[i_half -
                        step:] = numlib.arange(i, self._faces.shape[0])
            # from IPython import embed; embed()
            for rg in [rg0, rg1]:
                for j in range(3):
                    # rg = numlib.arange(
                    #     i, np.min((self._faces.shape[0], i + step)))
                    faces = allFaces[rg]
                    normTuple[faces[:, j],
                              vertexCount[faces[:, j]]] = faceNorms[rg]
                    vertexCount[faces[:, j]] += 1
        return normTuple
    # def reverseNormals(self):
        # """
        # Reverses the direction of all normal vectors.
        # """
        # pass

    # def generateEdgesFromFaces(self):
        # """
        # Generate a set of edges by listing all the edges of faces and
        # removing any duplicates.
        # Useful for displaying wireframe meshes.
        # """
        # pass

    def _computeEdges(self):
        if not self.hasFaceIndexedData():
            # generate self._edges from self._faces
            nf = len(self._faces)
            edges = np.empty(nf * 3, dtype=[('i', np.uint, 2)])
            edges['i'][0:nf] = self._faces[:, :2]
            edges['i'][nf:2 * nf] = self._faces[:, 1:3]
            edges['i'][-nf:, 0] = self._faces[:, 2]
            edges['i'][-nf:, 1] = self._faces[:, 0]

            # sort per-edge
            mask = edges['i'][:, 0] > edges['i'][:, 1]
            edges['i'][mask] = edges['i'][mask][:, ::-1]

            # remove duplicate entries
            self._edges = np.unique(edges)['i']
            # print self._edges
        elif self._vertexesIndexedByFaces is not None:
            verts = self._vertexesIndexedByFaces
            edges = np.empty((verts.shape[0], 3, 2), dtype=np.uint)
            nf = verts.shape[0]
            edges[:, 0, 0] = np.arange(nf) * 3
            edges[:, 0, 1] = edges[:, 0, 0] + 1
            edges[:, 1, 0] = edges[:, 0, 1]
            edges[:, 1, 1] = edges[:, 1, 0] + 1
            edges[:, 2, 0] = edges[:, 1, 1]
            edges[:, 2, 1] = edges[:, 0, 0]
            self._edges = edges
        else:
            raise Exception(
                "MeshData cannot generate edges--no faces in this data.")

    def save(self):
        """Serialize this mesh to a string appropriate for disk storage"""
        import pickle
        if self._faces is not None:
            names = ['_vertexes', '_faces']
        else:
            names = ['_vertexesIndexedByFaces']

        if self._vertexColors is not None:
            names.append('_vertexColors')
        elif self._vertexColorsIndexedByFaces is not None:
            names.append('_vertexColorsIndexedByFaces')

        if self._faceColors is not None:
            names.append('_faceColors')
        elif self._faceColorsIndexedByFaces is not None:
            names.append('_faceColorsIndexedByFaces')

        state = dict([(n, getattr(self, n)) for n in names])
        return pickle.dumps(state)

    def restore(self, state):
        """Restore the state of a mesh previously saved using save()"""
        import pickle
        state = pickle.loads(state)
        for k in state:
            if isinstance(state[k], list):
                if isinstance(state[k][0], QtGui.QVector3D):
                    state[k] = [[v.x(), v.y(), v.z()] for v in state[k]]
                state[k] = np.array(state[k])
            setattr(self, k, state[k])

    @staticmethod
    def sphere(rows, cols, radius=1.0, offset=True):
        """
        Return a MeshData instance with vertexes and faces computed
        for a spherical surface.
        """
        verts = np.empty((rows + 1, cols, 3), dtype=float)

        # compute vertexes
        phi = (np.arange(rows + 1) * np.pi / rows).reshape(rows + 1, 1)
        s = radius * np.sin(phi)
        verts[..., 2] = radius * np.cos(phi)
        th = ((np.arange(cols) * 2 * np.pi / cols).reshape(1, cols))
        if offset:
            # rotate each row by 1/2 column
            th = th + ((np.pi / cols) *
                       np.arange(rows + 1).reshape(rows + 1, 1))
        verts[..., 0] = s * np.cos(th)
        verts[..., 1] = s * np.sin(th)
        # remove redundant vertexes from top and bottom
        verts = verts.reshape((rows + 1) * cols, 3)[cols - 1:-(cols - 1)]

        # compute faces
        faces = np.empty((rows * cols * 2, 3), dtype=np.uint)
        rowtemplate1 = ((np.arange(cols).reshape(cols, 1) +
                        np.array([[0, 1, 0]])) % cols) + \
            np.array([[0, 0, cols]])
        rowtemplate2 = ((np.arange(cols).reshape(cols, 1) +
                         np.array([[0, 1, 1]])) % cols) + \
            np.array([[cols, 0, cols]])
        for row in range(rows):
            start = row * cols * 2
            faces[start:start + cols] = rowtemplate1 + row * cols
            faces[start + cols:start + (cols * 2)] = rowtemplate2 + row * cols
        # cut off zero-area triangles at top and bottom
        faces = faces[cols:-cols]

        # adjust for redundant vertexes that were removed from top and bottom
        vmin = cols - 1
        faces[faces < vmin] = vmin
        faces -= vmin
        vmax = verts.shape[0] - 1
        faces[faces > vmax] = vmax

        return MeshData(vertexes=verts, faces=faces)

    @staticmethod
    def cylinder(rows, cols, radius=[1.0, 1.0], length=1.0, offset=False):
        """
        Return a MeshData instance with vertexes and faces computed
        for a cylindrical surface.
        The cylinder may be tapered with different radii at each end
        (truncated cone)
        """
        verts = np.empty((rows + 1, cols, 3), dtype=float)
        if isinstance(radius, int):
            radius = [radius, radius]  # convert to list
        # compute vertexes
        th = np.linspace(2 * np.pi, (2 * np.pi) / cols, cols).reshape(1, cols)
        # radius as a function of z
        r = np.linspace(radius[0], radius[1], num=rows + 1,
                        endpoint=True).reshape(rows + 1, 1)
        verts[..., 2] = np.linspace(
            0, length, num=rows + 1, endpoint=True).reshape(rows + 1, 1)  # z
        if offset:
            # rotate each row by 1/2 column
            th = th + ((np.pi / cols) *
                       np.arange(rows + 1).reshape(rows + 1, 1))
        verts[..., 0] = r * np.cos(th)  # x = r cos(th)
        verts[..., 1] = r * np.sin(th)  # y = r sin(th)
        # just reshape: no redundant vertices...
        verts = verts.reshape((rows + 1) * cols, 3)
        # compute faces
        faces = np.empty((rows * cols * 2, 3), dtype=np.uint)
        rowtemplate1 = ((np.arange(cols).reshape(cols, 1) +
                        np.array([[0, 1, 0]])) % cols) + \
            np.array([[0, 0, cols]])
        rowtemplate2 = ((np.arange(cols).reshape(
            cols, 1) + np.array([[0, 1, 1]])) % cols) + \
            np.array([[cols, 0, cols]])
        for row in range(rows):
            start = row * cols * 2
            faces[start:start + cols] = rowtemplate1 + row * cols
            faces[start + cols:start + (cols * 2)] = rowtemplate2 + row * cols

        return MeshData(vertexes=verts, faces=faces)
