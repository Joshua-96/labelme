# from pyqtgraph.Qt import QtCore, QtGui
import matplotlib
import pyqtgraph as pg
import labelme.openglMod as gl
import numpy as np
import cv2


class render_3d(gl.GLViewWidget):
    def __init__(self, image: np.ndarray = None) -> None:
        super(gl.GLViewWidget, self).__init__()
        GL_ALPHA_TEST = gl.shaders.GL_ALPHA_TEST
        GL_CULL_FACE = gl.shaders.GL_CULL_FACE
        GL_CULL_FACE_MODE = gl.shaders.GL_CULL_FACE_MODE
        GL_BACK = gl.shaders.GL_BACK
        # gl.shaders.glFrustum
        glBlendFunc = gl.shaders.glBlendFunc
        GL_SRC_ALPHA = gl.shaders.GL_SRC_ALPHA
        GL_ONE_MINUS_SRC_ALPHA = gl.shaders.GL_ONE_MINUS_SRC_ALPHA
        gl.shaders.glEnable(GL_CULL_FACE)
        # gl.shaders.glEnable(gl.shaders.GL_AUTO_NORMAL)
        # gl.shaders.glDisable(GL_CULL_FACE)
        self.image = image
        if not image is None:
            self.preproc_img()


def draw_SurfacePlot(z_vals, z_scale=1, shader="shaded", smooth=False):
    # z_vals = z_vals.astype(np.uint8)
    # possible shader
    # balloon,viewNormalColor, shaded, edgeHilight, heightColor, normalColor
    from matplotlib import pyplot as plt
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(z_vals)

    if shader == "heightColor":
        computeNormals = False
    else:
        computeNormals = True
    Plot = gl.GLSurfacePlotItem(z=z_vals,
                                shader=shader,
                                computeNormals=computeNormals,
                                glOptions='opaque',
                                smooth=smooth,
                                # colors=rgba_img
                                )
    Plot.scale(1, 1, z_scale)
    Plot.updateGLOptions({gl.shaders.GL_CULL_FACE: False})
    Plot.translate(0, 0, 0)
    return Plot


def draw_grid(scale):
    grid = gl.GLGridItem()
    grid.scale(scale, scale, 1)
    # draw grid after surfaces since they may be translucent
    grid.setDepthValue(10)
    return grid
