import random

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.image import Image


class GenerateTab(QtWidgets.QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.dataset = []

        title = QtWidgets.QLabel("Generate Data")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        subtitle = QtWidgets.QLabel(
            "Create a dataset with X/O ratio and preview the first 4 samples."
        )
        subtitle.setStyleSheet("color: #a8b0c4;")

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.name_input = QtWidgets.QLineEdit("dataset_01")
        self.count_input = QtWidgets.QSpinBox()
        self.count_input.setRange(1, 10000)
        self.count_input.setValue(100)
        self.percent_x_input = QtWidgets.QSpinBox()
        self.percent_x_input.setRange(0, 100)
        self.percent_x_input.setValue(50)

        form.addRow("Dataset name:", self.name_input)
        form.addRow("Number of elements:", self.count_input)
        form.addRow("Percent X (0-100):", self.percent_x_input)

        self.generate_button = QtWidgets.QPushButton("Generate Dataset")
        self.generate_button.clicked.connect(self.generate_dataset)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: #9aa3ba;")

        self.datasets_label = QtWidgets.QLabel("Created datasets")
        self.datasets_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.datasets_list = QtWidgets.QListWidget()
        self.datasets_list.setFixedHeight(140)
        self.datasets_list.setStyleSheet(
            "background-color: #1f2430; border: 1px solid #3a4258; "
            "border-radius: 8px; color: #cfd6e6;"
        )
        self.datasets_list.currentRowChanged.connect(self._on_dataset_selected)
        self.app_state.datasets_changed.connect(self._refresh_datasets_list)
        self.app_state.active_dataset_changed.connect(self._on_active_dataset_changed)

        left_col = QtWidgets.QVBoxLayout()
        left_col.addWidget(title)
        left_col.addWidget(subtitle)
        left_col.addSpacing(12)
        left_col.addLayout(form)
        left_col.addSpacing(12)
        left_col.addWidget(self.generate_button)
        left_col.addSpacing(6)
        left_col.addWidget(self.status_label)
        left_col.addSpacing(10)
        left_col.addWidget(self.datasets_label)
        left_col.addSpacing(6)
        left_col.addWidget(self.datasets_list)
        left_col.addStretch(1)

        preview_title = QtWidgets.QLabel("Preview (first 4)")
        preview_title.setStyleSheet("font-size: 16px; font-weight: 600;")

        preview_grid = QtWidgets.QGridLayout()
        preview_grid.setSpacing(12)
        self.preview_labels = []
        for r in range(2):
            for c in range(2):
                label = QtWidgets.QLabel("No preview")
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setFixedSize(200, 200)
                label.setStyleSheet(
                    "background-color: #1f2430; border: 1px solid #3a4258; "
                    "border-radius: 8px; color: #8a93a8;"
                )
                preview_grid.addWidget(label, r, c)
                self.preview_labels.append(label)

        right_col = QtWidgets.QVBoxLayout()
        right_col.addWidget(preview_title)
        right_col.addLayout(preview_grid)
        right_col.addStretch(1)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left_col, 1)
        layout.addLayout(right_col, 2)

    def generate_dataset(self):
        name = self.name_input.text().strip() or "dataset"
        count = int(self.count_input.value())
        percent_x = int(self.percent_x_input.value())

        if count < 4:
            self.status_label.setText("Minimum dataset size is 4 (2 X and 2 O).")
            return

        count_x = int(round(count * (percent_x / 100.0)))
        count_o = count - count_x
        if count_x < 2 or count_o < 2:
            self.status_label.setText(
                "Need at least 2 X and 2 O. Adjust count or percentage."
            )
            return

        preview = self._show_preview_popup()
        if preview is None:
            self.status_label.setText("Preview cancelled.")
            return

        preview_x = sum(1 for item in preview if item["label"] == 1)
        preview_o = sum(1 for item in preview if item["label"] == 0)
        remaining_x = count_x - preview_x
        remaining_o = count_o - preview_o
        if remaining_x < 0 or remaining_o < 0:
            self.status_label.setText(
                "Preview exceeded target counts. Adjust count or percentage."
            )
            return

        self.dataset = list(preview)
        labels = [1] * remaining_x + [0] * remaining_o
        random.shuffle(labels)
        for label in labels:
            image = Image(size=40)
            if label == 1:
                image.draw_x()
            else:
                image.draw_o()
            record = {"pixels": np.array(image.pixels).flatten(), "label": label}
            self.dataset.append(record)

        self.status_label.setText(
            f"Created '{name}' with {count} items ({percent_x}% X / {100 - percent_x}% O)"
        )
        self._render_preview(preview)
        self.app_state.add_dataset(
            {
                "name": name,
                "count": count,
                "percent_x": percent_x,
                "data": self.dataset,
                "preview": preview,
            }
        )

    def _refresh_datasets_list(self):
        self.datasets_list.clear()
        for ds in self.app_state.datasets:
            self.datasets_list.addItem(
                f"{ds['name']}  |  {ds['count']} items  |  X {ds['percent_x']}% / O {100 - ds['percent_x']}%"
            )
        if self.app_state.active_dataset_index >= 0:
            self.datasets_list.setCurrentRow(self.app_state.active_dataset_index)

    def _on_dataset_selected(self, row):
        self.app_state.set_active_dataset(row)

    def _on_active_dataset_changed(self, row):
        if row < 0 or row >= len(self.app_state.datasets):
            return
        preview = self.app_state.datasets[row].get("preview", [])
        self._render_preview(preview)

    def _show_preview_popup(self):
        dialog = PreviewDialog(self)
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            return dialog.preview
        return None

    def _render_preview(self, preview):
        for i, label in enumerate(self.preview_labels):
            if i < len(preview):
                pixmap = self._pixels_to_pixmap(
                    preview[i]["pixels"], preview[i]["label"]
                )
                label.setPixmap(pixmap)
                label.setAlignment(QtCore.Qt.AlignCenter)
            else:
                label.setPixmap(QtGui.QPixmap())
                label.setText("No preview")

    def _pixels_to_pixmap(self, pixels, label_value):
        size = int(np.sqrt(len(pixels)))
        data = np.array(pixels, dtype=np.uint8).reshape((size, size))
        image = QtGui.QImage(
            data.data, size, size, size, QtGui.QImage.Format_Grayscale8
        )
        image = image.copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            180, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        label = "X" if label_value == 1 else "O"
        overlay = QtGui.QPixmap(pixmap.size())
        overlay.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(overlay)
        painter.drawPixmap(0, 0, pixmap)
        painter.setPen(QtGui.QColor(220, 230, 240))
        painter.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        painter.drawText(8, pixmap.height() - 8, f"Label: {label}")
        painter.end()
        return overlay


class PreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Creating Samples")
        self.setModal(True)
        self.resize(520, 420)

        self.preview = []
        self._queue = [1, 1, 0, 0]
        random.shuffle(self._queue)
        self._items = []
        for label_value in self._queue:
            image = Image(size=40)
            generator = image.iter_draw_x() if label_value == 1 else image.iter_draw_o()
            self._items.append(
                {"image": image, "label": label_value, "generator": generator}
            )

        title = QtWidgets.QLabel("Generating 2 X and 2 O samples...")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        subtitle = QtWidgets.QLabel("Watch the preview update in real time.")
        subtitle.setStyleSheet("color: #9aa3ba;")

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(12)
        self.labels = []
        for r in range(2):
            for c in range(2):
                label = QtWidgets.QLabel("...")
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setFixedSize(200, 160)
                label.setStyleSheet(
                    "background-color: #1f2430; border: 1px solid #3a4258; "
                    "border-radius: 8px; color: #8a93a8;"
                )
                grid.addWidget(label, r, c)
                self.labels.append(label)

        self.progress_label = QtWidgets.QLabel("0 / 4")
        self.progress_label.setStyleSheet("color: #9aa3ba;")

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addLayout(grid)
        layout.addSpacing(8)
        layout.addWidget(self.progress_label)
        layout.addLayout(buttons)

        self._step_index = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.start(20)

    def _step(self):
        if self._step_index >= len(self._items):
            self._timer.stop()
            self.accept()
            return

        item = self._items[self._step_index]
        try:
            next(item["generator"])
            self._update_label(self._step_index, item)
        except StopIteration:
            record = {
                "pixels": np.array(item["image"].pixels).flatten(),
                "label": item["label"],
            }
            self.preview.append(record)
            self._update_label(self._step_index, item, final=True)
            self._step_index += 1
            self.progress_label.setText(f"{self._step_index} / 4")
            if self._step_index >= len(self._items):
                self._timer.stop()
                self.accept()

    def _update_label(self, index, item, final=False):
        pixmap = self._pixels_to_pixmap(item["image"].pixels, item["label"])
        self.labels[index].setPixmap(pixmap)
        self.labels[index].setAlignment(QtCore.Qt.AlignCenter)

    def _pixels_to_pixmap(self, pixels, label_value):
        data = np.array(pixels, dtype=np.uint8)
        if data.ndim == 2:
            size = data.shape[0]
        else:
            size = int(np.sqrt(data.size))
            data = data.reshape((size, size))
        image = QtGui.QImage(
            data.data, size, size, size, QtGui.QImage.Format_Grayscale8
        )
        image = image.copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            180, 140, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        label = "X" if label_value == 1 else "O"
        overlay = QtGui.QPixmap(pixmap.size())
        overlay.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(overlay)
        painter.drawPixmap(0, 0, pixmap)
        painter.setPen(QtGui.QColor(220, 230, 240))
        painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        painter.drawText(6, pixmap.height() - 6, f"Label: {label}")
        painter.end()
        return overlay
