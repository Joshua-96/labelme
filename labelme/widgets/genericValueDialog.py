from msilib.schema import ComboBox
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QComboBox
from pyparsing import Combine
from qtpy.QtCore import Qt, Signal
from qtpy import QtWidgets
from functools import partial

class editVariablesDialog(QtWidgets.QDialog):
    
    updateAction = Signal(list)
    
    def __init__(self, parent=None,
                 defaultValues=None,
                 minValue=None,
                 maxValue=None,
                 highValueText="",
                 lowValueText="",
                 WindowTitle="generic Dialog",
                 helpText="generic Help Text",
                 reactive=False,
                 WindowWidth=400,
                 WindowHeight=150
                 ):
        super(editVariablesDialog, self).__init__(parent)

        self.minValue = minValue
        self.maxValue = maxValue
        self.highValueText = highValueText
        self.lowValueText = lowValueText
        self.helpText = helpText
        self.reactive = reactive

        self.setModal(True)
        self.setWindowTitle(WindowTitle)
        self.values = defaultValues
        self.sliders = []
        self.valueLabels = []
        self.smallEndLabel = []
        self.largeEndLabel = []
        for i, _ in enumerate(defaultValues):

            self.sliders.append(self._create_slider(minValue[i],
                                                    maxValue[i],
                                                    defaultValues[i],
                                                    i))
            self.valueLabels.append(QLabel())
            self.valueLabels[-1].setText(
                f"current Value {self.sliders[-1].value()}"
            )
            self.smallEndLabel.append(QLabel())
            self.smallEndLabel[-1].setText(self.lowValueText[i])
            self.largeEndLabel.append(QLabel())
            self.largeEndLabel[-1].setText(self.highValueText[i])
        if not self.reactive:
            self.applyButton = QPushButton()
            self.applyButton.setText("apply Change")
            self.applyButton.clicked.connect(self._apply_change)
            
        # self.setWhatsThis(self.helpText)

        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        self.setToolTip(self.helpText)

        self.parent = parent

        grid = QGridLayout()
        for i, slider in enumerate(self.sliders):
            grid.addWidget(self.smallEndLabel[i], 2 * i, 0, 1, 1)
            grid.addWidget(slider, 2 * i, 1, 1, 1)
            grid.addWidget(self.largeEndLabel[i], 2 * i, 2, 1, 1)
            grid.addWidget(self.valueLabels[i], 2 * i + 1, 1, 1, 1)
        if not self.reactive:
            grid.addWidget(self.applyButton, len(self.sliders) + 2, 1, 1, 1)
        self.setLayout(grid)
        self.setFixedSize(WindowWidth, WindowHeight)
        self.setSizeGripEnabled(False)

    def _create_slider(self, min_value, max_value, default_value, index):
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(min_value, max_value)
        slider.setValue(default_value)
        ValueChangeCallback = partial(self.onNewValue, index)
        slider.valueChanged.connect(ValueChangeCallback)
        return slider

    def onNewValue(self, element_index):
        self.valueLabels[element_index].setText(
            f"current Value {self.sliders[element_index].value()}"
        )
        if self.reactive:
            self._apply_change()

    def _apply_change(self):
        for i, slider in enumerate(self.sliders):
            self.values[i] = slider.value()
        self.updateAction.emit(self.values)
        if not self.reactive:
            self.close()


class DropdownDialog(QtWidgets.QDialog):
    """given a nested list of dropdown menus to show each with it's respective
        options, generates a Dialog with decription of the dropdown and
        of the options with OptionItem being a list of dictonaries with OptionName and description"""
    updateAction = Signal(list)

    def __init__(self, parent=None,
                 defaultValues: str = None,
                 OptionItems: list = None,
                 # OptionDescription = None,
                 descriptions: list = [],
                 WindowTitle="generic Dialog",
                 helpText="generic Help Text",
                 reactive=False,
                 WindowWidth=400,
                 WindowHeight=150
                 ):
        super(DropdownDialog, self).__init__(parent)

        self.defaultValues = defaultValues
        self.OptionItems = OptionItems
        self.descriptions = descriptions
        self.helpText = helpText
        self.reactive = reactive

        self.setModal(True)
        self.setWindowTitle(WindowTitle)
        self.values = defaultValues
        self.comboBox = []
        self.decriptionLabel = []
        self.OptionDescriptionLabel = []
        # self.OptionDescription = OptionDescription
        for i,dropDown in enumerate(OptionItems):
            self.comboBox.append(self.get_dropDown(dropDown, i))
            self.decriptionLabel.append(QLabel())
            self.decriptionLabel[-1].setText(self.descriptions[i])
            self.decriptionLabel[-1].setWordWrap(True)
            self.genericOptionDescription = QLabel()
            self.genericOptionDescription.setWordWrap(True)
            self.OptionDescriptionLabel.append(
                self.genericOptionDescription)
            self.comboBox[-1].setCurrentText(self.defaultValues[i])
            self.selectionChanged(i, init_call=True)
        # self.OptionDescriptionLabel.setText(self.OptionItems[0])

        if not self.reactive:
            self.applyButton = QPushButton()
            self.applyButton.setText("apply Change")
            self.applyButton.clicked.connect(self._apply_change)
            
        # self.setWhatsThis(self.helpText)

        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        self.setToolTip(self.helpText)

        self.parent = parent

        grid = QGridLayout()
        for i, cB in enumerate(self.comboBox):
            grid.addWidget(self.decriptionLabel[i], 2 * i, 0, 1, 1)
            grid.addWidget(cB, 2 * i, 1, 1, 1)
            grid.addWidget(self.OptionDescriptionLabel[i], 2 * i + 1, 1, 1, 1)
        if not self.reactive:
            grid.addWidget(self.applyButton, len(self.comboBox) + 2, 1, 1, 1)
        self.setLayout(grid)
        self.setFixedSize(WindowWidth, WindowHeight)
        self.setSizeGripEnabled(False)


    def selectionChanged(self, element_index, init_call=False):
        self.OptionDescriptionLabel[element_index].setText(
            self.OptionItems[element_index][
                self.comboBox[element_index].currentText()]
        )
        if self.reactive and not init_call:
            self._apply_change()

    def get_dropDown(self, dropDown, index):
        ComboBox = QComboBox()
        ComboBox.addItems(dropDown.keys())
        ValueChangeCallback = partial(self.selectionChanged, index)
        ComboBox.currentIndexChanged.connect(ValueChangeCallback)
        return ComboBox

    def _apply_change(self):
        for i, cB in enumerate(self.comboBox):
            self.values[i] = cB.currentText()
        # self.updateAction.emit(self.values)
        if not self.reactive:
            self.close()