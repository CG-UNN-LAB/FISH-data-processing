from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel


# Класс, необходимый для переопределения метода "mousePressEvent";
class ClickableLabel(QLabel):
    # Создаем новый сигнал;
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)

    def mousePressEvent(self, event):
        # Учет растяжения картинки в окне:
        pos = self.mapFromParent(event.position())
        label_size = self.size()
        scale_x = 512 / label_size.width()
        scale_y = 512 / label_size.height()
        original_x = pos.x() * scale_x
        original_y = pos.y() * scale_y
        # print(f"Координаты на исходном изображении: ({original_x}, {original_y})")

        # Излучаем сигнал при нажатии кнопки мыши;
        self.clicked.emit(original_x, original_y)
