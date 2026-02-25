import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.perceptron import Perceptron


class LearningTab(QtWidgets.QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.model = None
        self.training_data = []
        self.training_labels = np.array([])
        self.training_timer = QtCore.QTimer(self)
        self.training_timer.timeout.connect(self._train_step)
        self.current_step = 0
        self.total_steps = 0
        self.current_epoch = 0

        title = QtWidgets.QLabel("Learning")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        subtitle = QtWidgets.QLabel("Train perceptron in real time.")
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

        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setRange(1, 100000)
        self.epochs_input.setValue(5)
        self.images_per_second_input = QtWidgets.QSpinBox()
        self.images_per_second_input.setRange(1, 1000)
        self.images_per_second_input.setValue(50)

        self.start_button = QtWidgets.QPushButton("Start Learning")
        self.start_button.clicked.connect(self._start_learning)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Epochs:"))
        controls.addWidget(self.epochs_input)
        controls.addWidget(QtWidgets.QLabel("Images / second:"))
        controls.addWidget(self.images_per_second_input)
        controls.addWidget(self.start_button)
        controls.addStretch(1)

        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setStyleSheet("color: #9aa3ba;")

        visuals_frame = QtWidgets.QFrame()
        visuals_frame.setStyleSheet(
            "background-color: #1f2430; border: 1px solid #3a4258; "
            "border-radius: 8px; color: #8a93a8;"
        )
        visuals_layout = QtWidgets.QHBoxLayout(visuals_frame)
        visuals_layout.setSpacing(12)

        self.input_image_label = self._make_panel_label("Input image")
        self.weights_heatmap_label = self._make_panel_label("Weights heat map")
        self.output_label = self._make_panel_label("Output")
        self.output_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.output_label.setWordWrap(True)

        visuals_layout.addWidget(self.input_image_label, 2)
        visuals_layout.addWidget(self.weights_heatmap_label, 2)
        visuals_layout.addWidget(self.output_label, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(self.datasets_count_label)
        layout.addWidget(self.dataset_selector)
        layout.addWidget(self.active_dataset_label)
        layout.addSpacing(12)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        layout.addSpacing(8)
        layout.addWidget(visuals_frame, 1)

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

    def _make_panel_label(self, text):
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumHeight(280)
        label.setStyleSheet(
            "background-color: #171b24; border: 1px solid #3a4258; "
            "border-radius: 6px; color: #a8b0c4; padding: 8px;"
        )
        return label

    def _start_learning(self):
        active = self.app_state.get_active_dataset()
        if active is None:
            self.status_label.setText("No dataset selected.")
            return

        raw_data = active.get("data", [])
        if not raw_data:
            self.status_label.setText("Selected dataset is empty.")
            return

        self.training_data = [np.array(item["pixels"], dtype=float) for item in raw_data]
        self.training_labels = np.array([item["label"] for item in raw_data], dtype=float)
        n_weights = self.training_data[0].shape[0]
        self.model = Perceptron(n_weights=n_weights, learning_rate=0.01, random_state=0)

        epochs = int(self.epochs_input.value())
        self.total_steps = len(self.training_data) * epochs
        self.current_step = 0
        self.current_epoch = 1
        images_per_second = int(self.images_per_second_input.value())
        interval_ms = max(1, int(round(1000 / images_per_second)))

        self.start_button.setEnabled(False)
        self.epochs_input.setEnabled(False)
        self.images_per_second_input.setEnabled(False)
        self.status_label.setText(
            "Training started: "
            f"{epochs} epochs, {len(self.training_data)} images/epoch, "
            f"{images_per_second} images/second."
        )
        self.training_timer.start(interval_ms)

    def _train_step(self):
        if self.model is None or self.current_step >= self.total_steps:
            self._finish_training()
            return

        index = self.current_step % len(self.training_data)
        self.current_epoch = (self.current_step // len(self.training_data)) + 1

        x = self.training_data[index]
        y = self.training_labels[index]

        output = float(self.model.forward(x))
        error = float(y - output)
        self.model.weights += self.model.learning_rate * error * x
        self.model.bias += self.model.learning_rate * error

        self._render_input_image(x)
        self._render_weights_heatmap(self.model.weights)
        self._render_output(index, y, output, error)

        self.current_step += 1
        if self.current_step >= self.total_steps:
            self._finish_training()

    def _finish_training(self):
        self.training_timer.stop()
        self.start_button.setEnabled(True)
        self.epochs_input.setEnabled(True)
        self.images_per_second_input.setEnabled(True)
        if self.total_steps > 0 and self.current_step >= self.total_steps:
            self.status_label.setText("Training completed.")

    def _render_input_image(self, pixels):
        pixmap = self._array_to_grayscale_pixmap(pixels)
        self.input_image_label.setPixmap(pixmap)
        self.input_image_label.setAlignment(QtCore.Qt.AlignCenter)

    def _render_weights_heatmap(self, weights):
        heatmap = self._weights_to_heatmap(np.array(weights, dtype=float))
        self.weights_heatmap_label.setPixmap(heatmap)
        self.weights_heatmap_label.setAlignment(QtCore.Qt.AlignCenter)

    def _render_output(self, index, target, output, error):
        predicted = 1 if output >= 0.5 else 0
        text = (
            f"Epoch: {self.current_epoch}\n"
            f"Image: {index + 1}/{len(self.training_data)}\n"
            f"Target: {'X (1)' if int(target) == 1 else 'O (0)'}\n"
            f"Output: {output:.5f}\n"
            f"Predicted: {'X (1)' if predicted == 1 else 'O (0)'}\n"
            f"Error: {error:.5f}\n"
            f"Bias: {float(self.model.bias[0]):.5f}"
        )
        self.output_label.setText(text)
        self.output_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

    def _array_to_grayscale_pixmap(self, pixels):
        side = int(np.sqrt(pixels.size))
        data = np.array(pixels, dtype=np.uint8).reshape((side, side))
        image = QtGui.QImage(
            data.data, side, side, side, QtGui.QImage.Format_Grayscale8
        ).copy()
        return QtGui.QPixmap.fromImage(image).scaled(
            300, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

    def _weights_to_heatmap(self, weights):
        side = int(np.sqrt(weights.size))
        matrix = weights.reshape((side, side))
        min_w = float(np.min(matrix))
        max_w = float(np.max(matrix))

        rgb = np.full((side, side, 3), 255, dtype=np.uint8)
        red = np.array([235, 64, 52], dtype=float)
        blue = np.array([135, 206, 235], dtype=float)
        white = np.array([255, 255, 255], dtype=float)

        negative_mask = matrix < 0
        if min_w < 0:
            neg_ratio = np.zeros_like(matrix, dtype=float)
            neg_ratio[negative_mask] = matrix[negative_mask] / min_w
            neg_ratio = np.clip(neg_ratio, 0.0, 1.0)
            neg_colors = white + (red - white) * neg_ratio[..., None]
            rgb[negative_mask] = neg_colors[negative_mask].astype(np.uint8)

        positive_mask = matrix > 0
        if max_w > 0:
            pos_ratio = np.zeros_like(matrix, dtype=float)
            pos_ratio[positive_mask] = matrix[positive_mask] / max_w
            pos_ratio = np.clip(pos_ratio, 0.0, 1.0)
            pos_colors = white + (blue - white) * pos_ratio[..., None]
            rgb[positive_mask] = pos_colors[positive_mask].astype(np.uint8)

        image = QtGui.QImage(
            rgb.data, side, side, side * 3, QtGui.QImage.Format_RGB888
        ).copy()
        return QtGui.QPixmap.fromImage(image).scaled(
            300, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation
        )
