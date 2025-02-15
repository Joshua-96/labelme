import cv2
import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme import QT5
from labelme.shape import Shape
import labelme.utils


# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    chartUpdate = QtCore.Signal(list)
    cursorMoved = QtCore.Signal(QtCore.QPointF)
    selectionChanged = QtCore.Signal(list)
    requestNextImage = QtCore.Signal()
    requestPreviousImage = QtCore.Signal()
    UpdateRenderedShape = QtCore.Signal(Shape, int, bool)
    ViewPortSync = QtCore.Signal(QtGui.QWheelEvent)
    drawRenderedShape = QtCore.Signal(str)
    removeRenderedShape = QtCore.Signal(int)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1
    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click
                )
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        # lower value means finer contours and smaller distances between points
        self.trace_smothness = kwargs.pop("trace_smothness", 3)
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []  # type: list[Shape]
        self.shapesBackups = []
        self.current = None  # type: Shape
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []  # type: list[Shape]
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None  # type: Shape
        self.prevhShape = None  # type: Shape
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.tracingActive = False
        self.pause_tracing = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # self.beginShape = False
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.ZeroImg = None
        self.ImgDim = None
        self.distMap_crit = None
        self.keep_selected = False

    def init_poly_array(self):
        for s in self.shapes:
            s.poly_array = np.array([[p.x(), p.y()] for p in s.points])

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "trace",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()
        self.getDistMapUpdate()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.prevMovePoint = pos
        self.restoreCursor()
        if not self.pixmap.isNull():
            self.cursorMoved.emit(pos)
        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["polygon", "trace"]:
                self.line.shape_type = "polygon"
            else:
                self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip", "trace"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            if self.createMode == "trace":
                length = QtCore.QLineF(self.line[1], self.line[0]).length()
                if length > self.trace_smothness \
                        and int(ev.modifiers()) == QtCore.Qt.ShiftModifier:
                    self.current.addPoint(self.line[1])
                    self.UpdateRenderedShape.emit(self.current, -1, False)
                    self.line[0] = self.current[-1]
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons() and \
                int(ev.modifiers()) == QtCore.Qt.ControlModifier:
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [
                    s.copy() for s in self.selectedShapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint and \
                    int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            # else:
            #     raise ValueError
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        if not self.pixmap.isNull() and\
                ((pos.x() > 0 and pos.x() < self.imgDim[1])) and\
                (pos.y() > 0 and pos.y() < self.imgDim[0]):
            if int(ev.modifiers()) == QtCore.Qt.ShiftModifier:
                self.updateChart(pos)
            for i in range(len(self.shapes)):
                # Look for a nearby vertex to highlight. If that fails,
                # check if we happen to be inside a shape.
                if not self.distMap_crit[int(pos.y()), int(pos.x()), i] or\
                        not self.shapes[i].selected:
                    continue
                index, closest_vertex = self.shapes[i].nearestVertex(
                    pos,
                    self.epsilon / self.scale
                )
                if index is None:
                    index_edge = self.shapes[i].nearestEdge(
                        pos,
                        self.epsilon / self.scale,
                        minDistIndex=closest_vertex)
                else:
                    index_edge = None

                if index is not None:
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex = index
                    self.prevhShape = self.hShape = self.shapes[i]
                    self.prevhEdge = self.hEdge
                    self.hEdge = None
                    self.shapes[i].highlightVertex(
                        index,
                        self.shapes[i].MOVE_VERTEX
                    )
                    self.overrideCursor(CURSOR_POINT)
                    self.setToolTip(self.tr("Click & drag to move point"))
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break
                elif index_edge is not None and self.shapes[i].canAddPoint():
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex
                    self.hVertex = None
                    self.prevhShape = self.hShape = self.shapes[i]
                    self.prevhEdge = self.hEdge = index_edge
                    self.overrideCursor(CURSOR_POINT)
                    self.setToolTip(self.tr("Click to create point"))
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break
                elif self.shapes[i].containsPoint(pos):
                    if self.selectedVertex():
                        self.hShape.highlightClear()
                    self.prevhVertex = self.hVertex
                    self.hVertex = None
                    self.prevhShape = self.hShape = self.shapes[i]
                    self.prevhEdge = self.hEdge
                    self.hEdge = None
                    self.setToolTip(
                        self.tr(
                            "Click & drag to move shape '%s'"
                        ) % self.shapes[i].label
                    )
                    self.setStatusTip(self.toolTip())
                    if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                        self.overrideCursor(CURSOR_GRAB)
                    self.update()
                    break
            else:  # Nothing found, clear highlights, reset state.
                self.unHighlight()
            self.vertexSelected.emit(self.hVertex is not None)

    def apply_distTrans(self, shape, index):
        shape2draw = [[x[0] % self.ZeroImg.shape[1], x[1]]
                      for x in shape.poly_array]
        array_list = (np.array(shape2draw, dtype=np.int32).reshape(-1, 1, 2))
        binImg = cv2.drawContours(
            np.zeros([self.ZeroImg.shape[0], self.ZeroImg.shape[1], 3]),
            [array_list], -1, 1, -1
        )[:, :, 0]
        distMap = cv2.distanceTransform(
            cv2.bitwise_not((binImg * 255).astype(np.uint8)),
            cv2.DIST_L2,
            maskSize=0
        )
        # InvDistMap = cv2.distanceTransform(
        #     (binImg * 255).astype(np.uint8),
        #     cv2.DIST_L2,
        #     maskSize=0
        # )
        # InvDistMap = np.clip(InvDistMap, 0, 255).astype(np.uint8)
        distMap = np.clip(distMap, 0, 255).astype(np.uint8)
        self.distMap_crit[:, :, index] = (distMap <= (self.epsilon + 7)
                                          ).astype(np.bool)

    def init_zeroImg(self):
        self.ZeroImg = np.zeros(
            [self.imgDim[0],
             self.imgDim[1],
             len(self.shapes)
             ]
        )

    def getDistMapUpdate(self, index=None):
        if self.ZeroImg is None:
            self.ZeroImg = np.zeros(
                [self.imgDim[0],
                    self.imgDim[1],
                    len(self.shapes)
                 ]
            )
        if self.distMap_crit is None:
            self.distMap_crit = self.ZeroImg.astype(np.bool)
        if index is not None:
            if index >= self.distMap_crit.shape[-1]:
                # self.distMap_crit = self.ZeroImg.astype(np.bool)
                # if index >= self.distMap_crit.shape[-1]:

                self.ZeroImg = np.zeros(
                    [self.imgDim[0],
                        self.imgDim[1],
                        len(self.shapes)
                     ]
                )
        # FIXME change to elif statement and dstack to distMap_crit
        if index is None or index >= self.distMap_crit.shape[-1]:
            self.distMap_crit = self.ZeroImg.astype(np.bool)
            for i, s in enumerate(self.shapes):
                if not self.isVisible(s):
                    continue
                if not hasattr(s, "poly_array"):
                    s.poly_array = np.array([[p.x(), p.y()] for p in s.points])

                self.apply_distTrans(s, i)
        else:
            assert isinstance(index, int),\
                f"Index must be of Type int not {type(index)}"
            self.apply_distTrans(self.shapes[index], index)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex

        if shape is None or index is None:
            return
        shape.removePoint(index)
        # shape.poly_array = np.delete(shape.poly_array, shape.poly_array[index])
        shape_index = self.shapes.index(shape)

        self.UpdateRenderedShape.emit(shape, shape_index, True)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True

        # Save changes

        # self.getDistMapUpdate()

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        self.UpdateRenderedShape.emit(self.current, -1, False)
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode == "trace":
                        # self.mouseDoubleClickEvent(
                        #     QtCore.QEvent(QtCore.QEvent.MouseButtonDblClick)
                        # )
                        # self.tracingActive = False
                        pass
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.beginShape = True
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == "trace":
                        self.tracingActive = True
                    if self.createMode == "point":
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                    self.selectedVertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                selected_ind = np.argwhere(
                    np.array(
                        [s.selected for s in self.shapes]))
                if selected_ind.size > 0:
                    if self.distMap_crit[int(pos.y()), int(pos.x()), selected_ind].any():
                        self.keep_selected = True
                    else:
                        self.keep_selected = False
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None
                and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos
        elif ev.button() == QtCore.Qt.BackButton:
            self.requestPreviousImage.emit()
        elif ev.button() == QtCore.Qt.ForwardButton:
            self.requestNextImage.emit()


    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton and\
                self.current and self.tracingActive:
            if len(self.current.points) > 1:
                # self.mouseDoubleClickEvent(
                #     QtCore.QEvent(QtCore.QEvent.MouseButtonDblClick)
                # )
                # self.tracingActive = False
                pass
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if (
                not menu.exec_(self.mapToGlobal(ev.pos()))
                and self.selectedShapesCopy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (
                self.shapesBackups[-1][index].points
                != self.shapes[index].points
            ):
                self.storeShapes()
                self.shapeMoved.emit()
                self.getDistMapUpdate(index=index)

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        # self.getDistMapUpdate()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
            self.double_click == "close"
            and self.canCloseShape()
            and len(self.current) > 3
        ):
            self.current.popPoint()

            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        if not self.keep_selected:
            self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPoint(int(x1), int(y1)), QtCore.QPoint(
            int(x2),
            int(y2)
        )

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        # self.removeCurrentShape.emit()
        shape_index = self.shapes.index(shape)
        shape.moveVertexBy(index, pos - point)
        self.UpdateRenderedShape.emit(shape, shape_index, True)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(int(min(0, o1.x())), int(min(0, o1.y())))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(
                int(min(0, self.pixmap.width() - o2.x())),
                int(min(0, self.pixmap.height() - o2.y())),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
                shape_index = self.shapes.index(shape)
                self.UpdateRenderedShape.emit(shape, shape_index, True)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                deleted_shapes.append(shape)
                shape_index = self.shapes.index(shape)
                self.shapes.remove(shape)
                self.distMap_crit = np.delete(self.distMap_crit, shape_index, 2)
                self.removeRenderedShape.emit(shape_index)
            self.storeShapes()
            self.getDistMapUpdate()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)

        self.getDistMapUpdate()
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPoint(2, 2)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def updateChart(self, pos):
        pos_as_int = [int(pos.x()), int(pos.y())]
        # span = int(width * 1 / self.scale)
        self.chartUpdate.emit(pos_as_int)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        self.offset = self.offsetToCenter()
        p.translate(self.offset)
        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(
                shape
            ):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (
            self.fillDrawing()
            and self.createMode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(int(x), int(y))

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()
        index = len(self.shapes) - 1
        self.drawRenderedShape.emit("most_recent")
        self.getDistMapUpdate(index)

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(
                    int(x3),
                    int(min(max(0, y2), max(y3, y4)))
                )
            else:  # y3 == y4
                return QtCore.QPoint(
                    int(min(max(0, x2)), max(x3, x4)),
                    int(y3)
                )
        return QtCore.QPoint(int(x), int(y))

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPoint(int((x3 + x4) / 2), int((y3 + y4) / 2))
                d = labelme.utils.distance(m - QtCore.QPoint(int(x2), int(y2)))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            self.ViewPortSync.emit(ev)
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            elif QtCore.Qt.ShiftModifier == int(mods):
                self.scrollRequest.emit(
                    int(delta.y() / 4),
                    QtCore.Qt.Horizontal
                )
            else:
                # scroll
                self.scrollRequest.emit(
                    int(delta.x() / 4),
                    QtCore.Qt.Horizontal
                )
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(
                self.selectedShapes, self.prevPoint + offset
            )
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        # modifiers = ev.modifiers()
        key = ev.key()
        if key == QtCore.Qt.Key_Escape and self.current:
            self.current = None
            self.drawingPolygon.emit(False)
            # remove temporary rendered Shape in memory
            self.removeRenderedShape.emit(-1)
            self.update()
        elif key == QtCore.Qt.Key_Return and self.canCloseShape():
            self.finalise()
        elif key == QtCore.Qt.Key_F:
            # Hold down F key to avoid the cursor attracting to
            # the initial polygon point
            pass
            # self.pause_tracing = True

    # def keyReleaseEvent(self, ev):
    #     key = ev.key()
    #     if key == QtCore.Qt.Key_F and self.pause_tracing:
    #         self.pause_tracing = False
    #     elif key == QtCore.Qt.Key_F and not self.pause_tracing:
    #         self.pause_tracing = True

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()
