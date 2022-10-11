# from webbrowser import Error
from signal import Signals
import PIL.Image
import PIL.ImageEnhance
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QSpinBox
from qtpy.QtCore import Qt, Signal
from qtpy import QtGui
from qtpy import QtWidgets

import numpy as np
import cv2

from .. import utils


KERNEL_CORRECTION = {3: 500,
                     5: 200,
                     7: 50,
                     9: 0.3,
                     11: 0.1,
                     }

LOG_CLIPPING_RATE = 1.07

CLIPPING_CEIL = 2**18


class BrightnessContrastDialog(QtWidgets.QDialog):
    
    updateImage = Signal(np.ndarray)
    
    def __init__(self, img, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)

        self.setModal(True)
        self.setWindowTitle("Brightness/Contrast")

        self.slider_brightness = self._create_slider()
        self.slider_contrast = self._create_slider(range=[0, 150], default=0)
        self.apply_gauss_filter_checkbox = self._create_checkbox(
            self.connect_gauss_filter_Checkbox, "Gauss")
        self.apply_sobel_filter_checkbox = self._create_checkbox(
            self.connect_sobel_filter_Checkbox, "Sobel")
        self.normalize_pushButton = self._create_pushButton(
            self.call_normalize, "normalize")
        self.reset_pushButton = self._create_pushButton(
            self.reset, "reset processing")
        self.kernelSize = self._create_spinBox(
            min_val=3,
            max_val=11,
            step=2
        )
        self.kernelSize.valueChanged.connect(self.apply_sobel_filter)
        self.derivative = self._create_spinBox(
            min_val=1,
            max_val=3
        )
        self.derivative.valueChanged.connect(
            self.apply_sobel_filter)
        self.clip_level = self._create_spinBox(1, 100)
        self.clip_level.valueChanged.connect(self.apply_sobel_filter)
        self.clip_level.setValue(10)
        self.HasReset = True
        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.tr("Brightness"), self.slider_brightness)
        formLayout.addRow(
            self.tr("Contrast"),
            self.slider_contrast
        )
        formLayout.addRow(
            self.tr("Filter Options"),
            self.apply_gauss_filter_checkbox
        )
        formLayout.addRow(self.tr(""), self.apply_sobel_filter_checkbox)
        formLayout.addRow(self.tr("kernel size"), self.kernelSize)
        formLayout.addRow(self.tr("derivative size"), self.derivative)
        formLayout.addRow(self.tr("sobel z-scale"), self.clip_level)
        formLayout.addRow(self.tr(""), self.normalize_pushButton)
        formLayout.addRow(self.tr(""), self.reset_pushButton)

        self.setLayout(formLayout)

        assert isinstance(img, PIL.Image.Image)
        self.img = cv2.bitwise_not(np.array(img))
        self.callback = callback

    def brightness_contrast_transform(self):
        brightness = self.slider_brightness.value() - 50
        contrast = self.slider_contrast.value() / (3 * self.slider_contrast.maximum())
        img_np = self.img
        img_max = img_np.max()
        img_min = img_np.min()
        offset = contrast * img_max
        mod_max = int(img_max - offset)
        mod_min = int(img_min + offset)
        LUT = np.zeros(65536, dtype=np.uint16)
        LUT[mod_min: mod_max + 1] = np.linspace(start=0,
                                        stop=65535,
                                        num=(mod_max - mod_min) + 1,endpoint=True,
                                        dtype=np.uint16)
        # img = self.img
        # if img.mode != "L":
        #    img = img.convert("L")
        # img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        # img = PIL.ImageEnhance.Contrast(img).enhance(contrast)
        if brightness < 0:
            if img_np.dtype.name == "uint8":
                img_np = np.clip(cv2.subtract(img_np, -1 * brightness), 0, 255)
            elif img_np.dtype.name == "uint16":
                img_np = cv2.subtract(
                    img_np, -brightness * 255)
            else:
                pass

        if img_np.dtype.name == "uint8":
            img = (img_np + max(0, brightness)).astype(np.uint16)
            img = np.clip(img, 0, 255).astype(np.uint8)
        elif img_np.dtype.name == "uint16":
            img = (img_np + max(0, brightness * 255)
            ).astype(np.uint32)
            img = np.clip(img, 0, 2**16 - 1).astype(np.uint16)
            img = LUT[img]
        # img = cv2.convertScaleAbs(
        #     img_np,
        #     alpha=contrast,
        #     beta=max(0, brightness)
        # )

        self.HasReset = False
        return img

    def call_normalize(self):
        img = self.brightness_contrast_transform()
        img_t = utils.image.normalize_image(img)
        self.HasReset = False
        self.img = img_t
        self._apply_change(img_t)

    def reset(self):
        self.slider_brightness.setValue(50)
        self.slider_contrast.setValue(50)
        self.HasReset = True
        self.apply_sobel_filter_checkbox.setChecked(False)
        self.apply_gauss_filter_checkbox.setChecked(False)
        self._apply_change(self.img)

    @staticmethod
    def _create_pushButton(callback, buttonText):
        button = QtWidgets.QPushButton(buttonText)
        button.clicked.connect(callback)
        return button

    def onNewValue(self):
        img = self.brightness_contrast_transform()
        self.apply_sobel_filter_checkbox.setChecked(False)
        self.apply_gauss_filter_checkbox.setChecked(False)
        self._apply_change(img)

    def apply_gauss_filter(self):
        img = self.brightness_contrast_transform()
        img = cv2.GaussianBlur(img, (5, 5), sigmaX=3, sigmaY=3)
        self._apply_change(img)

    def connect_gauss_filter_Checkbox(self, checked):
        # checked = self.apply_gauss_filter_checkbox.isChecked()
        if checked:
            self.apply_gauss_filter()
        else:
            self.get_unprocessed_image()

    def connect_sobel_filter_Checkbox(self, checked):
        if checked:
            self.apply_sobel_filter()
        else:
            self.get_unprocessed_image()

    def apply_sobel_filter(self):
        if not self.apply_sobel_filter_checkbox.isChecked():
            return 0
        img = self.brightness_contrast_transform()
        try:
            img_x = cv2.Sobel(
                img,
                cv2.CV_64F,
                dx=self.derivative.value(),
                dy=0,
                ksize=self.kernelSize.value()
            )
            img_y = cv2.Sobel(
                img,
                cv2.CV_64F,
                dx=0,
                dy=self.derivative.value(),
                ksize=self.kernelSize.value()
            )
            img = np.add(img_x, img_y)
            img = np.clip(
                img,
                -LOG_CLIPPING_RATE**self.clip_level.value() /
                KERNEL_CORRECTION[self.kernelSize.value()] * CLIPPING_CEIL,

                LOG_CLIPPING_RATE**self.clip_level.value() /
                KERNEL_CORRECTION[self.kernelSize.value()] * CLIPPING_CEIL
            )

        except cv2.error:
            if self.kernelSize.value() % 2 == 0:
                self.errorMessage(
                    "image processing error",
                    f"kernel size of value {self.kernelSize.value()} is not valid,\
                        number must be uneven"
                )
            else:
                self.errorMessage(
                    "image processing error",
                    f"derivative of order {self.derivative.value()} can't\
                    be calculated with kernel size {self.kernelSize.value()}\
                    please change either parameter")
            return 0
        set_absolute = False
        if set_absolute:
            img = np.absolute(img)
        else:
            img = img - img.min()

        if self.img.dtype == "int32":
            img = (img * 2**16 - 1 / img.max()).astype(np.uint16)
        else:
            #img = (img * 255 / img.max()).astype(np.uint8)
            img = (img / img.max() * 2**16 - 1).astype(np.uint16)
        # img = utils.image.normalize_image(img)
        self._apply_change(img)
        return 1

    def get_unprocessed_image(self):
        img = self.brightness_contrast_transform()
        self._apply_change(img)

    def _create_slider(self, range=[0, 150],default=50):
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(range[0], range[1])
        slider.setValue(default)
        slider.valueChanged.connect(self.onNewValue)
        return slider

    def _create_checkbox(self, stateCallback, cbText):
        assert isinstance(cbText, str), "type is not string"
        checkbox = QtWidgets.QCheckBox(cbText)
        checkbox.stateChanged.connect(stateCallback)
        return checkbox

    def _create_spinBox(self, min_val, max_val, step=1):
        spinBox = QSpinBox()
        spinBox.setMaximum(max_val)
        spinBox.setMinimum(min_val)
        spinBox.setSingleStep(step)
        return spinBox

    def _apply_change(self, img):
        if isinstance(img, PIL.Image.Image):
            img_data = utils.img_pil_to_data(img)
            qimage = QtGui.QImage.fromData(img_data)
        elif isinstance(img, np.ndarray):
            self.updateImage.emit(img)

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )
