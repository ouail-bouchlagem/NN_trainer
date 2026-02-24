from PyQt5 import QtCore, QtWidgets


class LearningTab(QtWidgets.QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
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

        placeholder = QtWidgets.QLabel("Training controls will be placed here.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet(
            "background-color: #1f2430; border: 1px solid #3a4258; "
            "border-radius: 8px; color: #8a93a8; padding: 24px;"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(self.datasets_count_label)
        layout.addWidget(self.dataset_selector)
        layout.addWidget(self.active_dataset_label)
        layout.addSpacing(12)
        layout.addWidget(placeholder, 1)

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
