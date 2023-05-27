import io
import os
import shutil

import czifile
import numpy as np
from PIL import Image
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
from ultralytics import YOLO

from Application_Qt_module_ui import Ui_MainWindow


class Func(Ui_MainWindow):
    FILEPATH = "-1"
    FILENAME = "-1"
    FileModelPath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..\\Model\\my_yolov8_model_core_segmentation.pt",
    )
    model = YOLO(FileModelPath)

    def setupUiFunc(self, MainWindow):
        self.setupUi(MainWindow)
        self.add_functions()

    # Клик по кнопке -> (вызов функции):
    def add_functions(self):
        self.pushButtonStart.clicked.connect(self.add_image)
        self.pushButtonSeg.clicked.connect(self.predict_image)

    # Основная фун-я по сегментации:
    def predict_image(self):  # ИСПРАВИТЬ: не допускать точности, ниже 0.2;
        if Func.FILEPATH != "-1":  # Значит, что картинка была сохранена, путь есть;
            Accuracy = self.labelAccuracy.text()  # Берем точность с поля;
            # FileModelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\\Model\\my_yolov8_model_core_segmentation.pt")

            # model = YOLO(FileModelPath)

            img = Image.open(Func.FILEPATH)

            folder_path = (
                "..\\Photo_Qt\\" + Func.FILENAME
            )  # Имя папки. If ниже проверяет, существует ли такая папка, есла да - удалить.
            if os.path.exists(
                folder_path
            ):  # Фун-я model.predict не может сама перезаписать папку с полученным изображением;
                shutil.rmtree(folder_path)

            predictions = Func.model.predict(
                img,
                show=False,  # Не показывать результат дополнительным окном;
                classes=[0, 1],  # Есть два класса;
                save=True,  # Сохранять полученное фото;
                project="..\\Photo_Qt",  # Папка сохранения;
                name=Func.FILENAME,  # model.predict сначала создасть папку с именем Ui_MainWindow.FILENAME (внутри папки project = ' '), только затем положит туда изображнеие с таким же названием, как и img;
                show_labels=False,
                show_conf=False,
                save_txt=True,  # Сохранит рядом с обработанным изображением в папку Ui_MainWindow.FILENAME;
                conf=float(Accuracy),  # Точность;
                line_thickness=1,
            )  # Толщина границ выделения;
            Path = (
                "..\\Photo_Qt\\" + Func.FILENAME + "\\" + Func.FILENAME + ".jpg"
            )  # Берем путь, чтобы вывести на экран;
            pixmap = QPixmap(Path)
            self.label_2.setPixmap(pixmap)
            # Доп.Информация:
            names = Func.model.names
            number_whole = 0
            number_explode = 0
            for r in predictions:
                for c in r.boxes.cls:
                    if names[int(c)] == "Whole cell":
                        number_whole += 1
                    if names[int(c)] == "Explode cell":
                        number_explode += 1
            print("Explode:")
            print(number_explode)
            print("Whole:")
            print(number_whole)
            ref = (
                "Whole cell: "
                + str(number_whole)
                + "\nExplode cell: "
                + str(number_explode)
            )
            self.Reference.setText(ref)

    # фун-я, сохраняющая картинку:
    def PhotoSave(self, pixmap, filename):
        folder_name = "..\\Photo_Qt"  # Имя папки;

        if not os.path.exists(folder_name):  # Создать папку, если ее нет;
            os.makedirs(folder_name)

        Image_name = os.path.splitext(os.path.basename(filename))[
            0
        ]  # Для получения имени файла без расширения и пути к нему;

        dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..\\Photo_Qt"
        )
        jpg = ".jpg"
        # print (dir_path)
        filepath = os.path.join(
            dir_path, Image_name
        )  # Объединяет путь к директории (dir_path) и имя файла (Image_name), чтобы создать полный путь к файлу;
        filepath = filepath + jpg
        # print (filepath)
        Func.FILEPATH = filepath
        Func.FILENAME = Image_name
        # print (Ui_MainWindow.FILENAME)
        pixmap.save(filepath)

    # Обработка изображения, если оно передано в формате .czi:
    def process_image(self, filename):
        if filename.endswith(".czi"):
            # Откроем ZIP-файл с помощью czifile:
            with czifile.CziFile(os.path.join(filename)) as czi:
                # Получить все каналы в файле:
                channels = czi.asarray()[0, 0, :, :, :]
                # Объедините все каналы в единое RGB-изображение:
                image_array = np.stack([channels[1], channels[2], channels[0]], axis=-1)
                # Преобразуем изображение в Pillow:
                image_array = np.squeeze(image_array)
                image = np.uint8(image_array)
                img = Image.fromarray(image)
                img = img.resize((450, 450))
                # Преобразуем из Pillow в qimage через BytesIO:
                byte_array = io.BytesIO()
                img.save(byte_array, format="PNG")
                byte_array = byte_array.getvalue()
                qimage = QImage.fromData(byte_array)
                pixmap = QPixmap.fromImage(qimage)
                self.labelFoto.setPixmap(pixmap)
                # Создадим папку и положим туда выбранное изображение:
                self.PhotoSave(pixmap, filename)

    # Выбор изображения и добавление его в окно для исходного изображения:
    def add_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            None, "Open Image", os.getcwd(), "Images (*.png *.jpg *.czi)"
        )

        if filename:
            if filename.endswith(".jpg"):
                pixmap = QPixmap(filename).scaled(
                    450, 450, Qt.AspectRatioMode.KeepAspectRatio
                )  # window_size.width(), window_size.height(), Qt.AspectRatioMode.KeepAspectRatio
                self.labelFoto.setPixmap(pixmap)
                self.PhotoSave(pixmap, filename)

            if filename.endswith(".png"):
                pixmap = QPixmap(filename).scaled(
                    450, 450, Qt.AspectRatioMode.KeepAspectRatio
                )  # window_size.width(), window_size.height(), Qt.AspectRatioMode.KeepAspectRatio
                self.labelFoto.setPixmap(pixmap)
                self.PhotoSave(pixmap, filename)

            if filename.endswith(".czi"):
                self.process_image(filename)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Func()
    ui.setupUiFunc(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
