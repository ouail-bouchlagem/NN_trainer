from PyQt5 import QtCore, QtWidgets


class LearningTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        title = QtWidgets.QLabel("Learning")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        subtitle = QtWidgets.QLabel("Train perceptron in real time.")
        subtitle.setStyleSheet("color: #a8b0c4;")

        placeholder = QtWidgets.QLabel("Training controls will be placed here.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet(
            "background-color: #1f2430; border: 1px solid #3a4258; "
            "border-radius: 8px; color: #8a93a8; padding: 24px;"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(placeholder, 1)
