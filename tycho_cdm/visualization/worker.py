from PyQt5.QtCore import QObject, pyqtSignal


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.shouldClose = False

    def run(self):
        self.task()
