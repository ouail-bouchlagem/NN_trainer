import sys

from PyQt5 import QtGui, QtWidgets

from classes.app_state import AppState
from tabs.generate_tab import GenerateTab
from tabs.learning_tab import LearningTab
from tabs.testing_tab import TestingTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perceptron Trainer")
        self.resize(1200, 720)

        self.app_state = AppState(self)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(GenerateTab(self.app_state), "Generate Data")
        tabs.addTab(LearningTab(self.app_state), "Learning")
        tabs.addTab(TestingTab(self.app_state), "Testing")

        self.setCentralWidget(tabs)


def apply_theme(app):
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(20, 22, 28))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(230, 234, 242))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(28, 32, 42))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(32, 36, 46))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(230, 234, 242))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(230, 234, 242))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(230, 234, 242))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(32, 36, 46))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(230, 234, 242))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(80, 120, 180))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(10, 10, 10))
    app.setPalette(palette)


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
