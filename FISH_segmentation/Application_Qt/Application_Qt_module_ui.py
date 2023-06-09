# Form implementation generated from reading ui file 'Application_Qt.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(918, 684)
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("resources/icon.ico"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.Off,
        )
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(
            "#MainWindow\n"
            "{\n"
            "border-image: url(resources/Background4.jpg);\n"
            "background-position: center center no-repeat  fixed;\n"
            "background-size: 100% 100%;\n"
            "\n"
            "}\n"
            "\n"
            "\n"
            "\n"
            "\n"
            ""
        )
        MainWindow.setAnimated(True)
        MainWindow.setDocumentMode(False)
        MainWindow.setDockNestingEnabled(False)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setStyleSheet(
            "#centralwidget {background-color: transparent;}"
        )
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalGroupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.verticalGroupBox.setAcceptDrops(False)
        self.verticalGroupBox.setAutoFillBackground(False)
        self.verticalGroupBox.setStyleSheet(
            "#verticalGroupBox {\n"
            "border: 1px inset #10f04f;\n"
            "border-radius: 10px;\n"
            "}"
        )
        self.verticalGroupBox.setFlat(False)
        self.verticalGroupBox.setCheckable(False)
        self.verticalGroupBox.setObjectName("verticalGroupBox")
        self.vboxlayout = QtWidgets.QVBoxLayout(self.verticalGroupBox)
        self.vboxlayout.setObjectName("vboxlayout")
        self.horizontalWidget = QtWidgets.QWidget(parent=self.verticalGroupBox)
        self.horizontalWidget.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.NoContextMenu
        )
        self.horizontalWidget.setStyleSheet(
            "#horizontalGroupBox {\n" "border: 1px inset #10f04f;\n" "}"
        )
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.labelFoto = QtWidgets.QLabel(parent=self.horizontalWidget)
        self.labelFoto.setStyleSheet(
            "#labelFoto\n" "{\n" "border: 4px inset #10f0ad;\n" "\n" "}"
        )
        self.labelFoto.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.labelFoto.setObjectName("labelFoto")
        self.horizontalLayout.addWidget(self.labelFoto)
        self.label_2 = QtWidgets.QLabel(parent=self.horizontalWidget)
        self.label_2.setStyleSheet(
            "#label_2\n" "{\n" "border: 4px outset #10f0e9;\n" "\n" "}"
        )
        self.label_2.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.vboxlayout.addWidget(self.horizontalWidget)
        self.gridGroupBox = QtWidgets.QGroupBox(parent=self.verticalGroupBox)
        self.gridGroupBox.setMaximumSize(QtCore.QSize(16777215, 150))
        self.gridGroupBox.setStyleSheet(
            "#gridGroupBox {\n"
            "                width: 100px;\n"
            "                height:100px;\n"
            "                border: 3px solid #1071f0;\n"
            "                border-radius: 15px;\n"
            "}"
        )
        self.gridGroupBox.setObjectName("gridGroupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.gridGroupBox)
        self.gridLayout.setContentsMargins(9, 5, 9, 5)
        self.gridLayout.setHorizontalSpacing(9)
        self.gridLayout.setVerticalSpacing(7)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButtonSeg = QtWidgets.QPushButton(parent=self.gridGroupBox)
        self.pushButtonSeg.setStyleSheet(
            "QPushButton {\n"
            "  background-color: #ccaa2d;\n"
            "  color: white; \n"
            "  border: 1px solid gray;\n"
            "  padding: 15px;\n"
            "  border-radius: 10px;\n"
            "}\n"
            "\n"
            "QPushButton:pressed {\n"
            "    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
            "                                      stop: 0 #FF7832, stop: 1 #FF9739);\n"
            "}"
        )
        self.pushButtonSeg.setObjectName("pushButtonSeg")
        self.gridLayout.addWidget(self.pushButtonSeg, 1, 0, 1, 1)
        self.horizontalWidget_2 = QtWidgets.QWidget(parent=self.gridGroupBox)
        self.horizontalWidget_2.setMinimumSize(QtCore.QSize(0, 0))
        self.horizontalWidget_2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.horizontalWidget_2.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.PreventContextMenu
        )
        self.horizontalWidget_2.setLayoutDirection(
            QtCore.Qt.LayoutDirection.LeftToRight
        )
        self.horizontalWidget_2.setAutoFillBackground(False)
        self.horizontalWidget_2.setStyleSheet(
            "#horizontalGroupBox_2{           \n" "outline: none;\n" "}"
        )
        self.horizontalWidget_2.setObjectName("horizontalWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.labelStart = QtWidgets.QLabel(parent=self.horizontalWidget_2)
        self.labelStart.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.labelStart.setStyleSheet("#labelStart{ color: white; }")
        self.labelStart.setObjectName("labelStart")
        self.horizontalLayout_2.addWidget(self.labelStart)
        self.labelAccuracy = QtWidgets.QLineEdit(parent=self.horizontalWidget_2)
        self.labelAccuracy.setObjectName("labelAccuracy")
        self.horizontalLayout_2.addWidget(self.labelAccuracy)
        self.gridLayout.addWidget(self.horizontalWidget_2, 3, 0, 1, 1)
        self.pushButtonStart = QtWidgets.QPushButton(parent=self.gridGroupBox)
        self.pushButtonStart.setMinimumSize(QtCore.QSize(250, 0))
        self.pushButtonStart.setStyleSheet(
            "QPushButton {\n"
            "  background-color: #ccaa2d;\n"
            "  color: white; \n"
            "  border: 1px solid gray;\n"
            "  padding: 15px;\n"
            "  border-radius: 10px;\n"
            "}\n"
            "\n"
            "QPushButton:pressed {\n"
            "    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
            "                                      stop: 0 #FF7832, stop: 1 #FF9739);\n"
            "}"
        )
        self.pushButtonStart.setObjectName("pushButtonStart")
        self.gridLayout.addWidget(self.pushButtonStart, 0, 0, 1, 1)
        self.verticalGroupBox_2 = QtWidgets.QGroupBox(parent=self.gridGroupBox)
        self.verticalGroupBox_2.setMouseTracking(False)
        self.verticalGroupBox_2.setTabletTracking(False)
        self.verticalGroupBox_2.setAcceptDrops(False)
        self.verticalGroupBox_2.setAutoFillBackground(False)
        self.verticalGroupBox_2.setStyleSheet(
            "#verticalGroupBox_2 {\n"
            "border: 1px inset #1071f0;\n"
            "border-radius: 10px;\n"
            "}"
        )
        self.verticalGroupBox_2.setTitle("")
        self.verticalGroupBox_2.setFlat(False)
        self.verticalGroupBox_2.setCheckable(False)
        self.verticalGroupBox_2.setObjectName("verticalGroupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalGroupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Reference = QtWidgets.QLabel(parent=self.verticalGroupBox_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Reference.setFont(font)
        self.Reference.setAutoFillBackground(False)
        self.Reference.setStyleSheet("")
        self.Reference.setText("")
        self.Reference.setObjectName("Reference")
        self.verticalLayout_2.addWidget(self.Reference)
        self.gridLayout.addWidget(self.verticalGroupBox_2, 0, 1, 4, 1)
        self.vboxlayout.addWidget(self.gridGroupBox)
        self.verticalLayout.addWidget(self.verticalGroupBox)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FISH data processing"))
        self.labelFoto.setText(
            _translate(
                "MainWindow", "Исходное изображение, в формате .jpeg, png или .czi"
            )
        )
        self.label_2.setText(_translate("MainWindow", "Результат сегментации"))
        self.pushButtonSeg.setText(_translate("MainWindow", "Cегментировать"))
        self.labelStart.setText(_translate("MainWindow", "Порог точности:  "))
        self.labelAccuracy.setText(_translate("MainWindow", "0.9"))
        self.pushButtonStart.setText(_translate("MainWindow", "Выбрать фото"))
