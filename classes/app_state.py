from PyQt5 import QtCore


class AppState(QtCore.QObject):
    datasets_changed = QtCore.pyqtSignal()
    active_dataset_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets = []
        self.active_dataset_index = -1

    def add_dataset(self, dataset):
        self.datasets.append(dataset)
        self.datasets_changed.emit()
        self.set_active_dataset(len(self.datasets) - 1)

    def set_active_dataset(self, index):
        if not 0 <= index < len(self.datasets):
            return
        if self.active_dataset_index == index:
            return
        self.active_dataset_index = index
        self.active_dataset_changed.emit(index)

    def get_active_dataset(self):
        if 0 <= self.active_dataset_index < len(self.datasets):
            return self.datasets[self.active_dataset_index]
        return None
