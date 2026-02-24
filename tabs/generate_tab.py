import random

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from classes.image import Image


class GenerateTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = []
        self.datasets = []

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

        preview = []
        self.dataset = []
        for _ in range(count):
            image = Image(size=40)
            is_x = random.random() < (percent_x / 100.0)
            if is_x:
                image.draw_x()
                label = 1
            else:
                image.draw_o()
                label = 0
            record = {"pixels": np.array(image.pixels).flatten(), "label": label}
            self.dataset.append(record)
            if len(preview) < 4:
                preview.append(record)

        self.status_label.setText(
            f"Created '{name}' with {count} items ({percent_x}% X / {100 - percent_x}% O)"
        )
        self._render_preview(preview)
        self.datasets.append(
            {
                "name": name,
                "count": count,
                "percent_x": percent_x,
                "data": self.dataset,
                "preview": preview,
            }
        )
        self._refresh_datasets_list()
        self.datasets_list.setCurrentRow(len(self.datasets) - 1)

    def _refresh_datasets_list(self):
        self.datasets_list.clear()
        for ds in self.datasets:
            self.datasets_list.addItem(
                f"{ds['name']}  |  {ds['count']} items  |  X {ds['percent_x']}% / O {100 - ds['percent_x']}%"
            )

    def _on_dataset_selected(self, row):
        if row < 0 or row >= len(self.datasets):
            return
        preview = self.datasets[row].get("preview", [])
        self._render_preview(preview)

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
