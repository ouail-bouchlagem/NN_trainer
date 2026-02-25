import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.perceptron import Perceptron


class DrawingCanvas(QtWidgets.QLabel):
    def __init__(self, grid_size=40, display_size=320, parent=None):
        super().__init__(parent)
        self.grid_size = grid_size
        self.display_size = display_size
        self.pixel_size = max(1, self.display_size // self.grid_size)
        self.brush_radius = 2
        self._drawing = False

        self.buffer = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.setFixedSize(self.display_size, self.display_size)
        self.setStyleSheet(
            "background-color: #171b24; border: 1px solid #3a4258; border-radius: 8px;"
        )
        self._refresh_pixmap()

    def clear(self):
        self.buffer.fill(0)
        self._refresh_pixmap()

    def to_vector(self):
        return self.buffer.astype(float).flatten()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drawing = True
            self._paint_at(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._paint_at(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drawing = False
        super().mouseReleaseEvent(event)

    def _paint_at(self, pos):
        x = int(pos.x() / self.pixel_size)
        y = int(pos.y() / self.pixel_size)
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return

        for dy in range(-self.brush_radius, self.brush_radius + 1):
            for dx in range(-self.brush_radius, self.brush_radius + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if dx * dx + dy * dy <= self.brush_radius * self.brush_radius:
                        self.buffer[ny, nx] = 255

        self._refresh_pixmap()

    def _refresh_pixmap(self):
        image = QtGui.QImage(
            self.buffer.data,
            self.grid_size,
            self.grid_size,
            self.grid_size,
            QtGui.QImage.Format_Grayscale8,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self.display_size,
            self.display_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation,
        )
        self.setPixmap(pixmap)


class TestingTab(QtWidgets.QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.model = None

        title = QtWidgets.QLabel("Testing")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        subtitle = QtWidgets.QLabel("Draw your own sample and test it.")
        subtitle.setStyleSheet("color: #a8b0c4;")

        self.datasets_count_label = QtWidgets.QLabel("Datasets available: 0")
        self.datasets_count_label.setStyleSheet("color: #cfd6e6;")
        self.active_dataset_label = QtWidgets.QLabel("Selected dataset: none")
        self.active_dataset_label.setStyleSheet("color: #cfd6e6;")
        self.dataset_selector = QtWidgets.QComboBox()
        self.dataset_selector.setStyleSheet(
            "background-color: #1f2430; border: 1px solid #3a4258; "
            "border-radius: 6px; color: #cfd6e6; padding: 4px;"
        )
        self.dataset_selector.currentIndexChanged.connect(self._on_dataset_selected)

        controls_frame = QtWidgets.QFrame()
        controls_frame.setStyleSheet(
            "background-color: #1f2430; border: 1px solid #3a4258; "
            "border-radius: 8px; color: #8a93a8; padding: 12px;"
        )
        controls_layout = QtWidgets.QVBoxLayout(controls_frame)

        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setRange(1, 100000)
        self.epochs_input.setValue(5)
        self.train_button = QtWidgets.QPushButton("Train Model From Selected Dataset")
        self.train_button.clicked.connect(self._train_model)

        train_controls = QtWidgets.QHBoxLayout()
        train_controls.addWidget(QtWidgets.QLabel("Epochs:"))
        train_controls.addWidget(self.epochs_input)
        train_controls.addWidget(self.train_button)
        train_controls.addStretch(1)

        self.canvas = DrawingCanvas(grid_size=40, display_size=320)
        self.predict_button = QtWidgets.QPushButton("Predict Drawing")
        self.predict_button.clicked.connect(self._predict_drawing)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.canvas.clear)

        draw_controls = QtWidgets.QHBoxLayout()
        draw_controls.addWidget(self.predict_button)
        draw_controls.addWidget(self.clear_button)
        draw_controls.addStretch(1)

        self.status_label = QtWidgets.QLabel("Train a model, then draw with your mouse.")
        self.status_label.setStyleSheet("color: #9aa3ba;")
        self.result_label = QtWidgets.QLabel("Prediction: -")
        self.result_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #cfd6e6;")

        controls_layout.addLayout(train_controls)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.canvas, alignment=QtCore.Qt.AlignLeft)
        controls_layout.addLayout(draw_controls)
        controls_layout.addWidget(self.result_label)
        controls_layout.addWidget(self.status_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(self.datasets_count_label)
        layout.addWidget(self.dataset_selector)
        layout.addWidget(self.active_dataset_label)
        layout.addSpacing(12)
        layout.addWidget(controls_frame, 1)

        self.app_state.datasets_changed.connect(self._refresh_dataset_info)
        self.app_state.active_dataset_changed.connect(self._refresh_dataset_info)
        self._refresh_dataset_info()

    def _on_dataset_selected(self, index):
        self.app_state.set_active_dataset(index)

    def _refresh_dataset_info(self, *_):
        count = len(self.app_state.datasets)
        self.datasets_count_label.setText(f"Datasets available: {count}")

        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        for ds in self.app_state.datasets:
            self.dataset_selector.addItem(ds["name"])
        active_index = self.app_state.active_dataset_index
        if 0 <= active_index < count:
            self.dataset_selector.setCurrentIndex(active_index)
        self.dataset_selector.setEnabled(count > 0)
        self.dataset_selector.blockSignals(False)

        active = self.app_state.get_active_dataset()
        if active is None:
            self.active_dataset_label.setText("Selected dataset: none")
        else:
            self.active_dataset_label.setText(f"Selected dataset: {active['name']}")

    def _train_model(self):
        active = self.app_state.get_active_dataset()
        if active is None:
            self.status_label.setText("No dataset selected.")
            return

        raw_data = active.get("data", [])
        if not raw_data:
            self.status_label.setText("Selected dataset is empty.")
            return

        training_data = [np.array(item["pixels"], dtype=float) for item in raw_data]
        training_labels = np.array([item["label"] for item in raw_data], dtype=float)
        n_weights = training_data[0].shape[0]
        self.model = Perceptron(n_weights=n_weights, learning_rate=0.01, random_state=0)

        epochs = int(self.epochs_input.value())
        for _ in range(epochs):
            for index, x in enumerate(training_data):
                y = training_labels[index]
                output = float(self.model.forward(x))
                error = float(y - output)
                self.model.weights += self.model.learning_rate * error * x
                self.model.bias += self.model.learning_rate * error

        self.status_label.setText(
            f"Model trained on '{active['name']}' with {epochs} epoch(s)."
        )

    def _predict_drawing(self):
        if self.model is None:
            self.status_label.setText("Train the model first.")
            return

        x = self.canvas.to_vector()
        output = float(self.model.forward(x))
        predicted = "X (1)" if output >= 0.5 else "O (0)"
        self.result_label.setText(f"Prediction: {predicted}  |  Output: {output:.5f}")
        self.status_label.setText("Prediction completed.")
