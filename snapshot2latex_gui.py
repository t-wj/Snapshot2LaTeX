# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
import numpy as np
from predict import Predict

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 300)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.newFormula = QtWidgets.QPushButton(self.centralwidget)
        self.newFormula.setObjectName("newFormula")
        self.gridLayout_5.addWidget(self.newFormula, 0, 0, 1, 1)
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setObjectName("label3")
        self.gridLayout_5.addWidget(self.label3, 0, 1, 1, 1)
        self.shortcutSetting = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.shortcutSetting.setMinimumSize(QtCore.QSize(0, 1))
        self.shortcutSetting.setObjectName("shortcutSetting")
        self.gridLayout_5.addWidget(self.shortcutSetting, 0, 2, 1, 1)
        self.gridLayout_5.setColumnStretch(0, 6)
        self.gridLayout_5.setColumnStretch(1, 1)
        self.gridLayout_5.setColumnStretch(2, 2)
        self.gridLayout_6.addLayout(self.gridLayout_5, 2, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.result1 = QtWidgets.QTextEdit(self.centralwidget)
        self.result1.setMinimumSize(QtCore.QSize(0, 1))
        self.result1.setObjectName("result1")
        self.gridLayout.addWidget(self.result1, 0, 1, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setObjectName("label1")
        self.gridLayout_2.addWidget(self.label1, 0, 0, 1, 1)
        self.copy1 = QtWidgets.QPushButton(self.centralwidget)
        self.copy1.setObjectName("copy1")
        self.gridLayout_2.addWidget(self.copy1, 1, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.result_img1 = QtWidgets.QGridLayout()
        self.result_img1.setObjectName("result_img1")
        self.gridLayout.addLayout(self.result_img1, 0, 2, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 10)
        self.gridLayout.setColumnStretch(2, 10)
        self.gridLayout_6.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.img = QtWidgets.QGridLayout()
        self.img.setObjectName("img")
        self.gridLayout_6.addLayout(self.img, 0, 0, 1, 1)
        self.gridLayout_6.setRowStretch(0, 4)
        self.gridLayout_6.setRowStretch(1, 2)
        self.gridLayout_6.setRowStretch(2, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Snapshot-to-LaTeX converter"))
        self.newFormula.setText(_translate("MainWindow", "New Formula"))
        self.label3.setText(_translate("MainWindow", "Shortcut"))
        self.shortcutSetting.setPlainText(
            _translate("MainWindow", "Alt+M"))
        self.label1.setText(_translate("MainWindow", "Result"))
        self.copy1.setText(_translate("MainWindow", "Copy"))

class SnapLabel(QtWidgets.QLabel):
    returnTrigger = QtCore.pyqtSignal()
    snap = None

    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)
        main = QtWidgets.QVBoxLayout(self)
        self.selection = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, self)

    def mouseReleaseEvent(self, event):
        self.returnTrigger.emit()
        self.snap = self.pixmap().copy(self.selection.geometry()).toImage(
        ).convertToFormat(QtGui.QImage.Format_Grayscale8)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.origin = QtCore.QPoint(event.pos())
            self.selection.show()

    def mouseMoveEvent(self, event):
        if self.selection.isVisible():
            self.selection.setGeometry(QtCore.QRect(
                self.origin, event.pos()).normalized())


class SnapWindow(QtWidgets.QDialog):
    def __init__(self):
        super(SnapWindow, self).__init__()
        QtWidgets.QMainWindow.__init__(self)

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = SnapLabel(self)
        pixmap = QtGui.QScreen.grabWindow(
            app.primaryScreen(), app.desktop().winId())
        gray = pixmap.toImage().convertToFormat(QtGui.QImage.Format_Grayscale8)
        self.label.setPixmap(QtGui.QPixmap.fromImage(gray))
        layout.addWidget(self.label)

        self.label.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocusProxy(self.label)
        self.label.setFocus(True)

        self.setLayout(layout)

        geometry = app.desktop().availableGeometry()
        self.setGeometry(geometry)

        self.label.returnTrigger.connect(self.returnSnap)
        self.label.returnTrigger.connect(self.close)
        self.show()

    def returnSnap(self):
        return self.label.snap


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setShortcut = lambda: self.newFormula.setShortcut(
            QtGui.QKeySequence(self.shortcutSetting.toPlainText()))
        self.setShortcut()

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        self.addMpl(self.img, fig)

        self.newFormula.clicked.connect(self.setImg)
        self.flushImg1 = lambda: self.flushImg(
            self.result1.toPlainText(), self.result_img1)
        self.result1.textChanged.connect(self.flushImg1)
        self.copyResult1 = lambda: self.copyResult(self.result1.toPlainText(), 1)
        self.shortcutSetting.textChanged.connect(self.setShortcut)
        self.copy1.clicked.connect(self.copyResult1)

        self.predict = Predict()

    def addMpl(self, layout, fig):
        while layout.count() > 0:
            layout.takeAt(0).widget().deleteLater()
  
        canvas = FigureCanvas(fig)
        fig.tight_layout(pad=0)
        layout.addWidget(canvas)
        canvas.draw()

    def setImg(self):
        self.hide()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)

        snapwindow = SnapWindow()
        snapwindow.setWindowModality(QtCore.Qt.ApplicationModal)
        snapwindow.exec_()

        QtWidgets.QApplication.restoreOverrideCursor()
        self.show()
        if snapwindow.result() == 0:
            qimage = snapwindow.returnSnap()
            if qimage is not None:
                img_raw = np.frombuffer(qimage.bits().asstring(qimage.height() * qimage.bytesPerLine(
                )), dtype=np.uint8).reshape((qimage.height(), qimage.bytesPerLine()))[:, :qimage.width()]

                # img_raw = np.array([[QtGui.qRed(qimage.pixel(x, y)) for x in range(
                #     qimage.width())] for y in range(qimage.height())], dtype=np.uint8)
               
                fig = Figure()
                ax = fig.add_subplot(111)
                ax.imshow(img_raw, interpolation='gaussian', cmap=plt.cm.gray)
                ax.axis('off')
                self.addMpl(self.img, fig)

                self.statusBar().showMessage('Processing...')
                try:
                    self.result1.setPlainText('')
                    self.copy1.setText('Copy')
                    self.copy1.setStyleSheet('')
                    latex = self.predict.predict_img(img_raw)
                    self.result1.setPlainText(latex)
                    self.copyResult1()
                except:
                    latex = ''
                    del self.predict
                    self.predict = Predict()
                self.statusBar().clearMessage()


    def flushImg(self, latex, layout):
        try:
            fig = Figure()
            ax = fig.add_subplot(111)
            plt.style.use('classic')
            ax.text(0, 0.5, '${} $'.format(latex), fontsize=20)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            [s.set_visible(False) for s in ax.spines.values()]
            self.addMpl(layout, fig)
        except:
            pass

    def copyResult(self, str, idx):
        QtWidgets.QApplication.clipboard().setText(str)
        if idx == 1:
            self.copy1.setText('Copied')
            self.copy1.setStyleSheet('font: italic')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    splash = QtWidgets.QSplashScreen(QtGui.QPixmap(
        500, 300), QtCore.Qt.WindowStaysOnTopHint)
    label = QtWidgets.QLabel("Snapshot-to-LaTeX\n\nConverter", splash)
    label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    label.setStyleSheet('color: white; font: 20pt \"Consolas\"')
    label.setGeometry(0, 0, 500, 300)
    splash.show()

    mainWin = MainWindow()
    mainWin.show()
    splash.finish(mainWin)
    sys.exit(app.exec_())
