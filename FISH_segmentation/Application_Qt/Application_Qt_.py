import os
import czifile
import numpy as np
import matplotlib.image as mpimg
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QTableWidgetItem, QMessageBox
from ultralytics import YOLO
from Application_Qt_module_ui import Ui_MainWindow
from ChromosomePatch import ChromosomeCellDetector
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class Func(Ui_MainWindow):
    width = -1
    height = -1

    SelectionListIndex = ""
    SelectionListIndexProm = ""
    ImagesDictionary = {}
    SegmentationImagesDictionary = {}

    # он же ref;
    ResultsDictionary = {}
    ModelDetectorsDictionary = {}

    AXList = []

    FileModelPath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..\\Model\\my_yolov8_model_core_segmentation_plus_plus.pt",
    )
    model = YOLO(FileModelPath)

    def setupUiFunc(self, MainWindow):
        self.setupUi(MainWindow)
        self.add_functions()

    # Клик по кнопке -> (вызов функции):
    def add_functions(self):
        self.pushButtonStart.clicked.connect(self.Add_Paths)
        self.pushButtonSeg.clicked.connect(self.ClickPushButtonSeg)
        self.pushButtonSave.clicked.connect(self.SavePhoto)
        self.SelectionList.clicked.connect(self.SelectionListFunc)
        self.SelectionTable.clicked.connect(self.SelectionTableFunc)

        self.deleteShortcut = QtGui.QShortcut(QtCore.Qt.Key.Key_Delete, self.SelectionList)
        self.deleteShortcut.activated.connect(self.ListFuncDelete)
        self.AccuracySlider.valueChanged.connect(self.update_label)

    # Обновить значение точности в поле на интерфейсе;
    def update_label(self):
        value = float(self.AccuracySlider.value() / 100)
        self.AccuracyLable.setText(str(value))

    # Функция для работы с таблицей;
    def SelectionTableFunc(self):
        if self.SelectionListIndexProm != "":
            selected_items = self.SelectionTable.selectedItems()
            if len(selected_items) > 0:
                row = selected_items[0].row()
            try:
                detector = self.ModelDetectorsDictionary[self.SelectionListIndexProm]
                radius = detector.Radius[row]

                item = self.SelectionTable.item(row, 1)
                if item is not None:
                    x = item.text()
                item = self.SelectionTable.item(row, 2)
                if item is not None:
                    y = item.text()
                image = self.SegmentationImagesDictionary[self.SelectionListIndexProm]
                desired_size = (512, 512)
                image = cv2.resize(image, desired_size)

                # Нарисовать круг вокруг выбранных координат
                cv2.circle(image, (int(float(x)), int(float(y))), int(float(radius)), (231, 242, 12), 2)

                # Отобразить изображение с кругом
                qimage = QImage(image, int(512), int(512), QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio)
                self.PlaceForPromFotos.setPixmap(pixmap)

            except (UnboundLocalError, IndexError) as e:
                error_dialog = QMessageBox()
                error_dialog.setWindowTitle("Ошибка")
                error_dialog.setText(f"Произошла ошибка: {str(e)}. Попробуйте перевыбрать элемент в таблице.")
                error_dialog.exec()

                print("Ошибка таблицы.")

    def ClickPushButtonSeg(self):
        is_checked = self.checkBoxSeg.isChecked()
        if (self.SelectionListIndex == "" and not is_checked) or len(self.ImagesDictionary) == 0:
            return
        Accuracy = float(self.AccuracySlider.value() / 100)  # Берем точность с поля;
        if (Accuracy >= 0.98):
            Accuracy = 0.95
        if is_checked:
            for ImageName in self.ImagesDictionary:
                self.predict_image_ChromosomePatch(Accuracy, ImageName)
        else:
            self.predict_image_ChromosomePatch(Accuracy, self.SelectionListIndex)

    # Функция для сегментации:
    def predict_image_ChromosomePatch(self, Accuracy, ImageName):
        try:
            detectorTest = ChromosomeCellDetector(self.ImagesDictionary[ImageName])
            detector = ChromosomeCellDetector(self.ImagesDictionary[ImageName])
            number_explode, number_whole = detector.find_cells(Accuracy)
            detector.detect_chromosomes()
            Red_Chromosome = detector.RedChromosome
            Green_Chromosome = detector.GreenChromosome

            fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
            ax = detector.plot(ax)
            fig.patch.set_visible(False)
            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            canvas = FigureCanvas(fig)
            canvas.draw()
            self.width, self.height = fig.get_size_inches() * fig.get_dpi()
            buffer_rgba = canvas.buffer_rgba()
            array_rgba = np.asarray(buffer_rgba)
            imgrgd = detectorTest.rgba2rgb(array_rgba)

            self.SegmentationImagesDictionary[ImageName] = imgrgd
            self.ModelDetectorsDictionary[ImageName] = detector

            ref = (
                "Whole cell: "
                + str(number_whole)
                + "\nExplode cell: "
                + str(number_explode)
                + "\nRed chromosome: "
                + str(Red_Chromosome)
                + "\nGreen chromosome: "
                + str(Green_Chromosome)
            )
            self.ResultsDictionary[ImageName] = ref
            detector.write_to_csv("Списочек", "рядом", ImageName)

            qimage = QImage(imgrgd, int(self.width), int(self.height), QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio)

            self.PlaceForPromFotos.setPixmap(pixmap)
            self.DataLabel.setText(ref)
        except IndexError as e:
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Ошибка")
            error_dialog.setText(f"Произошла ошибка: {str(e)}. Попробуйте увеличить точность.")
            error_dialog.exec()

            print("Ошибка. Попробуйте увеличить точность.")

    # фун-я, читающая картинки:
    def PhotoReadAndSave(self, FilePaths):
        file = 0
        folder_name = "..\\Photo_Qt"  # Имя папки;
        if not os.path.exists(folder_name):  # Создать папку, если ее нет;
            os.makedirs(folder_name)

        for file in range(0, len(FilePaths)):
            ImageName = os.path.splitext(os.path.basename(FilePaths[file]))[0]  # Для получения имени файла
            FindImageInList = self.SelectionList.findItems(ImageName, QtCore.Qt.MatchFlag.MatchExactly)
            if not FindImageInList:
                if FilePaths[file].endswith(".jpg"):
                    image = mpimg.imread(FilePaths[file])
                    image = np.ascontiguousarray(image)

                if FilePaths[file].endswith(".png"):
                    image = mpimg.imread(FilePaths[file])
                    image = (255 * image).astype(np.uint8)  # normalize the data to 0-255
                    image = np.ascontiguousarray(image)

                if FilePaths[file].endswith(".czi"):
                    image = self.read_czi_image(FilePaths[file])

                if np.all(image == 0):
                    break
                image = cv2.resize(image, (512, 512))
                detectorTest = ChromosomeCellDetector(image)
                image = detectorTest.rgba2rgb(image)

                self.ImagesDictionary[ImageName] = image
                self.SelectionList.addItem(ImageName)

    def read_czi_image(self, filename, norm=True):
        try:
            with czifile.CziFile(filename) as czi:
                image = czi.asarray().squeeze()
                image = np.stack([image[1], image[2], image[0]], axis=-1)  # swap channels to order -> RGB

                if norm:
                    info = np.iinfo(image.dtype)
                    image = image.astype(np.float64) / info.max  # normalize the data to 0-1
                    image = (255 * image).astype(np.uint8)  # normalize the data to 0-255

                    image = np.ascontiguousarray(image)
                return image
        except IndexError as e:
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Ошибка")
            error_dialog.setText(f"Произошла ошибка: {str(e)}. Проблема с обработкой .czi формата.")
            error_dialog.exec()
            print("Ошибка. Проблема с обработкой .czi формата.")
            return np.zeros((3, 3))

    # Получение массива путей к изображениям:
    def Add_Paths(self):
        FilePaths, _ = QFileDialog.getOpenFileNames(None, "Open Image", os.getcwd(), "Images (*.png *.jpg *.czi)")
        if FilePaths:
            self.PhotoReadAndSave(FilePaths)

    # Обработка взаимодействия со списком выбранных изображений "SelectionList":
    def SelectionListFunc(self):
        self.SelectionListIndex = (self.SelectionList.currentItem()).text()
        SegTrueIndex = ""
        for name in self.SegmentationImagesDictionary:
            if name == self.SelectionListIndex:
                SegTrueIndex = name

        myImage = Image.fromarray(self.ImagesDictionary[self.SelectionListIndex])
        # Преобразование PIL Image в QImage
        MyQImage = QImage(myImage.tobytes(), myImage.size[0], myImage.size[1], QImage.Format.Format_RGB888)
        # Преобразование QImage в QPixmap
        MyQPixmap = QPixmap.fromImage(MyQImage)
        self.PlaceForFotos.setPixmap(MyQPixmap)

        if SegTrueIndex != "":
            myQImageSeg = QImage(self.SegmentationImagesDictionary[SegTrueIndex], int(self.width), int(self.height),
                                 QImage.Format.Format_RGB888)
            MyQPixmapSeg = QPixmap.fromImage(myQImageSeg)
            MyQPixmapSeg = MyQPixmapSeg.scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.PlaceForPromFotos.setPixmap(MyQPixmapSeg)

            self.DataLabel.setText(self.ResultsDictionary[SegTrueIndex])
        else:
            self.PlaceForPromFotos.clear()
            self.DataLabel.clear()
            self.SelectionTable.setRowCount(0)

        # Работа с таблицей:
        self.SelectionListIndexProm = SegTrueIndex
        if SegTrueIndex != "":
            self.SelectionTable.clearContents()
            detector = self.ModelDetectorsDictionary[SegTrueIndex]
            self.SelectionTable.setRowCount(detector.NumberWhole + detector.NumberExplode)
            index = 0
            for idx, cell in enumerate(detector.cells):
                for center_of_mass in cell.center_of_mass:
                    self.SelectionTable.setItem(index, 0, QTableWidgetItem(str(idx + 1)))
                    self.SelectionTable.setItem(index, 1, QTableWidgetItem(str(center_of_mass[1])))
                    self.SelectionTable.setItem(index, 2, QTableWidgetItem(str(center_of_mass[0])))
                    self.SelectionTable.setItem(index, 3, QTableWidgetItem(str(len(cell.green_chromosomes))))
                    self.SelectionTable.setItem(index, 4, QTableWidgetItem(str(len(cell.red_chromosomes))))
                    self.SelectionTable.setItem(index, 5, QTableWidgetItem("Exploded" if cell.Type == 0 else "Whole"))
                    index += 1

    def ListFuncDelete(self):
        if self.SelectionList.count() == 0:
            return
        currentRow = self.SelectionList.currentRow()
        CurrentName = (self.SelectionList.currentItem()).text()
        self.SelectionList.takeItem(currentRow)

        self.ImagesDictionary.pop(CurrentName)
        if CurrentName in self.SegmentationImagesDictionary:
            self.SegmentationImagesDictionary.pop(CurrentName)
            self.ResultsDictionary.pop(CurrentName)
            self.ModelDetectorsDictionary.pop(CurrentName)
            self.SelectionTable.setRowCount(0)
            self.PlaceForPromFotos.clear()
        self.PlaceForFotos.clear()

    def SavePhoto(self):
        is_checked = self.checkBoxSave.isChecked()
        if is_checked:
            Folder_Path = QFileDialog.getExistingDirectory(
                None, "Select a folder:", "", QFileDialog.Option.ShowDirsOnly)
            if Folder_Path:
                for name in self.ImagesDictionary:
                    plt.imsave(
                        Folder_Path + "\\" + name + ".png",
                        self.ImagesDictionary[name])

                if not os.path.exists(Folder_Path + "\\PhotoSeg"):
                    os.makedirs(Folder_Path + "\\PhotoSeg")
                for name in self.SegmentationImagesDictionary:
                    plt.imsave(Folder_Path + "\\PhotoSeg" + "\\"
                               + name + ".png", self.SegmentationImagesDictionary[name])

        else:
            currentRow = self.SelectionList.currentRow()
            if currentRow != -1:
                Folder_Path = QFileDialog.getExistingDirectory(None, "Select a folder:", "",
                                                               QFileDialog.Option.ShowDirsOnly)
                if Folder_Path:
                    selected_item = self.SelectionList.currentItem()
                    selected_text = selected_item.text()
                    plt.imsave(Folder_Path + "\\" + selected_text + ".png", self.ImagesDictionary[selected_text])

                    if selected_text in self.SegmentationImagesDictionary:
                        if not os.path.exists(Folder_Path + "\\PhotoSeg"):
                            os.makedirs(Folder_Path + "\\PhotoSeg")
                        plt.imsave(Folder_Path + "\\PhotoSeg" + "\\" + selected_text + ".png",
                                   self.SegmentationImagesDictionary[selected_text])


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Func()
    ui.setupUiFunc(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
