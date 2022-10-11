import pyqtgraph.opengl as gl
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import getConfigOption

class cust_GLViewWidget(gl.GLViewWidget):
    """custom Class in order to overwrite the default camera controls"""

    requestPreviousImage = QtCore.Signal()
    requestNextImage = QtCore.Signal()

    def __init__(self,
                 parent=None,
                 devicePixelRatio=None,
                 rotationMethod='euler'):
        super().__init__(parent, devicePixelRatio, rotationMethod)
        
        
    def reset(self, opts=None):
        if opts is None:
            # will always appear at the center of the widget
            self.opts['center'] = QtGui.QVector3D(2500, 0, 0)  
            # camera rotation (quaternion:wxyz)
            self.opts['rotation'] = QtGui.QQuaternion(1,0,0,0) 
            self.opts['distance'] = 2500 # distance of camera from center
            self.opts['fov'] = 60       # horizontal field of view in degrees
            self.opts['elevation'] = 30 # camera's angle of elevation in degrees
            self.opts['azimuth'] = 20   # camera's azimuthal angle in degrees
        else:
            self.opts = opts
        self.setBackgroundColor(getConfigOption('background'))

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        diff = lpos - self.mousePos
        self.mousePos = lpos
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            # if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            #     self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            # else:
            self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
            self.pan(0, 0, diff.y(), relative='view-upright')
        

    def mousePressEvent(self, ev):
        
        if ev.button() == QtCore.Qt.BackButton:
            self.requestPreviousImage.emit()
        elif ev.button() == QtCore.Qt.ForwardButton:
            self.requestNextImage.emit()
        
        return super().mousePressEvent(ev)

    def wheelEvent(self, ev, diff_x=None, diff_y=None):
        """
        takes in a wheel event and applies the respective View Transformation
        optionally with a diff_x and diff_y parameter indicating a specfic off
        to pan the view, otherwise use the angle delta of Mousewheel
        """
        delta = ev.angleDelta()
        if diff_x is None:
            diff_x = ev.angleDelta().x()
        if diff_y is None:
            diff_y = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            # self.opts['fov'] *= 0.999**delta
            self.opts['distance'] *= 0.999**delta.y()
        elif (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
            # self.opts['distance'] *= 0.999**delta
            # TODO add frame of reference as a option to GUI
            self.pan(0, -diff_y, 0, relative='global')
        else:
            self.pan(-diff_y, -diff_x, 0, relative='global')
        self.update()


