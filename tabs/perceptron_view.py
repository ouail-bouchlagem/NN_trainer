import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QPen, QColor

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.perceptron import Perceptron

class PerceptronVisualizer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Perceptron Visualizer")

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.button = QPushButton("Train Step")
        self.button.clicked.connect(self.train_step)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.button)
        self.setLayout(layout)

        # Data
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 1, 1, 1])  # OR gate

        self.model = Perceptron(n_weights=2, learning_rate=0.1)

        self.current_index = 0

        self.draw_network()

    def draw_network(self):
        self.scene.clear()

        # Positions
        self.input_positions = [(50, 100), (50, 200)]
        self.output_position = (300, 150)

        # Draw inputs
        self.input_nodes = []
        for pos in self.input_positions:
            node = QGraphicsEllipseItem(0, 0, 40, 40)
            node.setPos(*pos)
            node.setBrush(QBrush(Qt.gray))
            self.scene.addItem(node)
            self.input_nodes.append(node)

        # Draw output
        self.output_node = QGraphicsEllipseItem(0, 0, 50, 50)
        self.output_node.setPos(*self.output_position)
        self.output_node.setBrush(QBrush(Qt.darkGray))
        self.scene.addItem(self.output_node)

        # Draw connections
        self.lines = []
        for i, pos in enumerate(self.input_positions):
            line = QGraphicsLineItem(
                pos[0] + 40,
                pos[1] + 20,
                self.output_position[0],
                self.output_position[1] + 25,
            )
            self.scene.addItem(line)
            self.lines.append(line)

        self.update_visuals()

    def update_visuals(self):
        # Update weights visualization
        for i, line in enumerate(self.lines):
            w = self.model.weights[i]

            thickness = 1 + abs(w) * 10
            color = QColor(0, 200, 0) if w >= 0 else QColor(200, 0, 0)

            pen = QPen(color)
            pen.setWidthF(thickness)
            line.setPen(pen)

        # Update output activation brightness
        x = self.X[self.current_index]
        output = self.model.forward(x)

        intensity = int(float(output) * 255)
        self.output_node.setBrush(QBrush(QColor(intensity, intensity, 0)))

    def train_step(self):
        x = self.X[self.current_index]
        y = self.y[self.current_index]

        output = self.model.forward(x)
        error = y - output

        self.model.weights += self.model.learning_rate * error * x
        self.model.bias += self.model.learning_rate * error

        self.update_visuals()

        self.current_index = (self.current_index + 1) % len(self.X)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PerceptronVisualizer()
    window.resize(500, 400)
    window.show()
    sys.exit(app.exec_())
