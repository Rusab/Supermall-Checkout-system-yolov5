# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'New ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1144, 851)
        
        palette = QPalette()
        palette.setColor(QPalette.Text, QtCore.Qt.white)
        
        
        myFont=QtGui.QFont('Poppins')
        myFont.setBold(True)
        myFont.setPointSize(16)
        
        titleFont=QtGui.QFont('Poppins')
        titleFont.setBold(True)
        titleFont.setPointSize(30)
        
        MainWindow.setStyleSheet("""
        QMainWindow {
            background-color: rgb(37, 39, 40);
        }
        
        QMenuBar{
            background-color: rgb(30, 32, 33);
        }
        
        QMenuBar::item::selected {
            background-color: #000000;
        }
        
        QMenu {
            background-color: #3b3e3f;
        }
        
        QMenu::item::selected {
            background-color: #000000;
        }
                                 
                                 """)
                                 
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.camFeed = QtWidgets.QLabel(self.centralwidget)
        self.camFeed.setGeometry(QtCore.QRect(50, 170, 640, 480))
        self.camFeed.setText("")
        self.camFeed.setPixmap(QtGui.QPixmap("ui_elements/connectingcam.jpg"))
        self.camFeed.setObjectName("camFeed")
        
        self.labelCam = QtWidgets.QLabel(self.centralwidget)
        self.labelCam.setGeometry(QtCore.QRect(50, 125, 151, 41))
        self.labelCam.setObjectName("labelCam")
        self.labelCam.setFont(myFont)
        self.labelCam.setPalette(palette)
        self.labelCam.setText("Camera Feed")
        self.labelCam.setStyleSheet("color:#ffffff")
        
        
        
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(720, 170, 241, 481))
        self.listWidget.setStyleSheet("background-color: rgb(59, 62, 63);")
        self.listWidget.setObjectName("listWidget")
        
        self.listPrice = QtWidgets.QListWidget(self.centralwidget)
        self.listPrice.setGeometry(QtCore.QRect(970, 170, 121, 481))
        self.listPrice.setStyleSheet("background-color: rgb(59, 62, 63);")
        self.listPrice.setObjectName("listPrice")
        
        self.listLabel = QtWidgets.QLabel(self.centralwidget)
        self.listLabel.setGeometry(QtCore.QRect(720, 125, 151, 41))
        self.listLabel.setObjectName("listLabel")
        self.listLabel.setText("Product List")
        self.listLabel.setFont(myFont)
        self.listLabel.setStyleSheet("color:#ffffff")
        
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 20, 631, 111))
        self.label_4.setText("Automated Billing System")
        self.label_4.setObjectName("label_4")
        self.label_4.setFont(titleFont)
        self.label_4.setStyleSheet("color:#5aff6d")

        self.versionLabel = QtWidgets.QLabel(self.centralwidget)
        self.versionLabel.setGeometry(QtCore.QRect(760, 60, 151, 41))
        self.versionLabel.setObjectName("versionLabel")
        self.versionLabel.setText("v.1.0")
        self.versionLabel.setFont(myFont)
        self.versionLabel.setStyleSheet("color:#ffffff")
        
        self.billingButton = QtWidgets.QPushButton(self.centralwidget)
        self.billingButton.setGeometry(QtCore.QRect(80, 670, 181, 61))
        self.billingButton.setStyleSheet("background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        
        
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui_elements/startbilling.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.billingButton.setIcon(icon)
        self.billingButton.setIconSize(QtCore.QSize(27, 27))
        self.billingButton.setObjectName("billingButton")
        self.lockButton = QtWidgets.QPushButton(self.centralwidget)
        self.lockButton.setGeometry(QtCore.QRect(280, 670, 181, 61))
        self.lockButton.setStyleSheet("background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("ui_elements/lockitems.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lockButton.setIcon(icon1)
        self.lockButton.setIconSize(QtCore.QSize(27, 27))
        self.lockButton.setObjectName("lockButton")
        self.resetButton = QtWidgets.QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QtCore.QRect(480, 670, 181, 61))
        self.resetButton.setStyleSheet("background-color: rgb(90, 255, 109);\n"
"font: 500 12pt \"Poppins\";")
        
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("ui_elements/clearbill.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.resetButton.setIcon(icon2)
        self.resetButton.setIconSize(QtCore.QSize(27, 27))
        self.resetButton.setObjectName("resetButton")
        self.totalLabel = QtWidgets.QLabel(self.centralwidget)
        self.totalLabel.setGeometry(QtCore.QRect(760, 670, 151, 61))
        self.totalLabel.setText("Total: " +"\u09F3" +"0")
        self.totalLabel.setFont(myFont)
        self.totalLabel.setStyleSheet("color:#ffffff")
        self.totalLabel.setObjectName("totalLabel")
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1144, 21))
        
       
       
        self.menubar.setPalette(palette)
        self.menubar.setStyleSheet("color:#ffffff;")
        self.menubar.setNativeMenuBar(True)
        self.menubar.setObjectName("menubar")
        self.menuFIle = QtWidgets.QMenu(self.menubar)
        self.menuFIle.setObjectName("menuFIle")
        self.menuFIle.setStyleSheet("background-color:#3b3e3f")
        
        self.menuAction = QtWidgets.QMenu(self.menubar)
        self.menuAction.setObjectName("menuAction")
        self.menuAction.setStyleSheet("background-color:#3b3e3f")
        
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuHelp.setStyleSheet("background-color:#3b3e3f")
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.actionOpen_Price_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_Price_File.setObjectName("actionOpen_Price_File")
        
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        
        self.actionStart_Billing = QtWidgets.QAction(MainWindow)
        self.actionStart_Billing.setObjectName("actionStart_Billing")
        
        self.actionStop_Billing = QtWidgets.QAction(MainWindow)
        self.actionStop_Billing.setObjectName("actionStop_Billing")
        
        self.actionLock_Items = QtWidgets.QAction(MainWindow)
        self.actionLock_Items.setObjectName("actionLock_Items")
        
        self.actionClear_List = QtWidgets.QAction(MainWindow)
        self.actionClear_List.setObjectName("actionClear_List")
        
        self.actionHow_to_use = QtWidgets.QAction(MainWindow)
        self.actionHow_to_use.setObjectName("actionHow_to_use")
        
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        
        self.menuFIle.addAction(self.actionOpen_Price_File)
        self.menuFIle.addAction(self.actionExit)
        
        self.menuAction.addAction(self.actionStart_Billing)
        self.menuAction.addAction(self.actionStop_Billing)
        self.menuAction.addAction(self.actionLock_Items)
        self.menuAction.addAction(self.actionClear_List)
        
        self.menuHelp.addAction(self.actionHow_to_use)
        self.menuHelp.addAction(self.actionAbout)
        
        self.menubar.addAction(self.menuFIle.menuAction())
        self.menubar.addAction(self.menuAction.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Automated Billing System v.1.0"))
        self.billingButton.setText(_translate("MainWindow", "  START BILLING"))
        self.lockButton.setText(_translate("MainWindow", "  LOCK ITEMS"))
        self.resetButton.setText(_translate("MainWindow", "  CLEAR LIST"))
        #self.totalLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-family:\'poppins\'; font-size:16pt; font-weight:600; color:#ffffff;\">Total: </span></p></body></html>"))
        self.menuFIle.setTitle(_translate("MainWindow", "FIle"))
        self.menuAction.setTitle(_translate("MainWindow", "Action"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen_Price_File.setText(_translate("MainWindow", "Open Price File"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionStart_Billing.setText(_translate("MainWindow", "Start Billing"))
        self.actionStart_Billing.setShortcut(_translate("MainWindow", "S"))
        self.actionStop_Billing.setText(_translate("MainWindow", "Stop Billing"))
        self.actionLock_Items.setText(_translate("MainWindow", "Lock Items"))
        self.actionLock_Items.setShortcut(_translate("MainWindow", "Space"))
        self.actionClear_List.setText(_translate("MainWindow", "Clear List"))
        self.actionClear_List.setShortcut(_translate("MainWindow", "X"))
        self.actionHow_to_use.setText(_translate("MainWindow", "How to use"))
        self.actionAbout.setText(_translate("MainWindow", "About"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())