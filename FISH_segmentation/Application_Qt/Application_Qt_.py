import os
import czifile
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import pyplot as plt
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
from ultralytics import YOLO
from Application_Qt_module_ui import Ui_MainWindow
from ChromosomePatch import ChromosomeCellDetector
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class Func(Ui_MainWindow):
    width = -1
    height = -1
    PhotoList = []
    PhotoNameList = []
    PhotoSegmentationList = []
    PhotoSegmentationNameList = []
    SelectionListIndex = -1
    Ref = []
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
        self.pushButtonSeg.clicked.connect(self.predict_image_ChromosomePatch)
        self.pushButtonSave.clicked.connect(self.SavePhoto)
        self.SelectionList.clicked.connect(self.SelectionListFunc)

        self.deleteShortcut = QtGui.QShortcut(QtCore.Qt.Key.Key_Delete, self.SelectionList)
        self.deleteShortcut.activated.connect(self.ListFuncDelete)
    
    # Функция для сегментации:
    def predict_image_ChromosomePatch(self):
        is_checked = self.checkBoxSeg.isChecked()
        if ((self.SelectionListIndex == -1 and not is_checked) or len(self.PhotoList) == 0):
            return
        Accuracy = float(self.labelAccuracy.text())  # Берем точность с поля;

        if is_checked:
            Zero = 0
            Index = len(self.PhotoList)
        else:
            Index = self.SelectionListIndex + 1
            Zero = self.SelectionListIndex
        for file in range(Zero, Index):
            detectorTest = ChromosomeCellDetector(self.PhotoList[file])
            # img = detectorTest.rgba2rgb(self.PhotoList[self.SelectionListIndex])
            detector = ChromosomeCellDetector(self.PhotoList[file])
            number_explode, number_whole = detector.find_cells(Accuracy)
            detector.detect_chromosomes()
            RedChromosome = detector.RedChromosome
            GreenChromosome = detector.GreenChromosome

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
            self.PhotoSegmentationList.append(imgrgd)
            self.PhotoSegmentationNameList.append(file)
            ref = (
                "Whole cell: "
                + str(number_whole)
                + "\nExplode cell: "
                + str(number_explode)
                + "\nRed chromosome: "
                + str(RedChromosome)
                + "\nGreen chromosome: "
                + str(GreenChromosome)
            )
            self.Ref.append(ref)
            detector.write_to_csv("Списочек", "рядом", self.PhotoNameList[file])
            
        qimage = QImage(imgrgd, int(self.width), int(self.height), QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio)
        self.PlaceForPromFotos.setPixmap(pixmap)
        self.DataLabel.setText(ref)

    # фун-я, читающая картинки:
    def PhotoReadAndSave(self, FilePaths):
        file = 0
        folder_name = "..\\Photo_Qt"  # Имя папки;
        if not os.path.exists(folder_name):  # Создать папку, если ее нет;
            os.makedirs(folder_name)

        for file in range(0, len(FilePaths)):
            Image_name = os.path.splitext(os.path.basename(FilePaths[file]))[0]  # Для получения имени файла
            items = self.SelectionList.findItems(Image_name, QtCore.Qt.MatchFlag.MatchExactly)
            if not items:
                if FilePaths[file].endswith(".jpg"):
                    image = mpimg.imread(FilePaths[file])
                    image = np.ascontiguousarray(image)

                if FilePaths[file].endswith(".png"):
                    image = mpimg.imread(FilePaths[file])
                    image = (255 * image).astype(np.uint8)  # normalize the data to 0-255
                    image = np.ascontiguousarray(image)

                if FilePaths[file].endswith(".czi"):
                    image = self.read_czi_image(FilePaths[file])

                detectorTest = ChromosomeCellDetector(image)
                image = detectorTest.rgba2rgb(image)

                self.PhotoNameList.append(Image_name)
                self.PhotoList.append(image)
                self.SelectionList.addItem(Image_name)

    def read_czi_image(self, filename, norm=True):
        with czifile.CziFile(filename) as czi:
            image = czi.asarray().squeeze()
            image = np.stack([image[1], image[2], image[0]], axis=-1)  # swap channels to order -> RGB

            if norm:
                info = np.iinfo(image.dtype)
                image = image.astype(np.float64) / info.max  # normalize the data to 0-1
                image = (255 * image).astype(np.uint8)  # normalize the data to 0-255

                image = np.ascontiguousarray(image)
            return image
        
    # Получение массива путей к изображениям:
    def Add_Paths(self):
        FilePaths, _ = QFileDialog.getOpenFileNames(None, "Open Image", os.getcwd(), "Images (*.png *.jpg *.czi)")
        if FilePaths:
            self.PhotoReadAndSave(FilePaths)

    def SelectionListFunc(self):
        self.SelectionListIndex = self.SelectionList.currentRow()

        Index = -1
        for name in range(0, len(self.PhotoSegmentationNameList)):
            if (self.PhotoSegmentationNameList[name] == self.SelectionListIndex):
                Index = name

        img = Image.fromarray(self.PhotoList[self.SelectionListIndex])
        # Преобразование PIL Image в QImage
        qim = QImage(img.tobytes(), img.size[0], img.size[1], QImage.Format.Format_RGB888)
        # Преобразование QImage в QPixmap
        pix = QPixmap.fromImage(qim)
        self.PlaceForFotos.setPixmap(pix)
        if (Index != -1):
            qimage = QImage(self.PhotoSegmentationList[Index], int(self.width), int(self.height),
                            QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.PlaceForPromFotos.setPixmap(pixmap)
            self.DataLabel.setText(self.Ref[Index])
        else:
            self.PlaceForPromFotos.clear()
            self.DataLabel.clear()

    def ListFuncDelete(self):
        currentRow = self.SelectionList.currentRow()
        self.SelectionList.takeItem(currentRow)
        self.PhotoNameList.pop(currentRow)
        self.PhotoList.pop(currentRow)
        Index = -1
        for name in range(0, len(self.PhotoSegmentationNameList)):
            if (self.PhotoSegmentationNameList[name] == currentRow):
                Index = name
        if (Index != -1):
            self.PhotoSegmentationList.pop(Index)
            self.PhotoSegmentationNameList.pop(Index)
            self.Ref.pop(Index)
        for name in range(0, len(self.PhotoSegmentationNameList)):
            if (self.PhotoSegmentationNameList[name] >= currentRow):
                self.PhotoSegmentationNameList[name] = self.PhotoSegmentationNameList[name] - 1

    def SavePhoto(self):
        is_checked = self.checkBoxSave.isChecked()
        if is_checked:
            Folder_Path = QFileDialog.getExistingDirectory(None, 'Select a folder:', '', QFileDialog.Option.ShowDirsOnly)
            if Folder_Path:
                for name in range(0, len(self.PhotoList)):
                    plt.imsave(Folder_Path + '\\' + self.PhotoNameList[name] + '.png', self.PhotoList[name])
                
                if not os.path.exists(Folder_Path + "\\PhotoSeg"):
                    os.makedirs(Folder_Path + "\\PhotoSeg")
                for name in range(0, len(self.PhotoSegmentationNameList)):
                    plt.imsave(Folder_Path + "\\PhotoSeg" + '\\' + self.PhotoNameList[self.PhotoSegmentationNameList[name]] + '.png', self.PhotoSegmentationList[name])

        else:
            currentRow = self.SelectionList.currentRow()
            if (currentRow != -1):
                Folder_Path = QFileDialog.getExistingDirectory(None, 'Select a folder:', '', QFileDialog.Option.ShowDirsOnly)
                if Folder_Path:
                    selected_item = self.SelectionList.currentItem()
                    selected_text = selected_item.text()
                    plt.imsave(Folder_Path + '\\' + selected_text + '.png', self.PhotoList[currentRow])
                    
                    Index = -1
                    for name in range(0, len(self.PhotoSegmentationNameList)):
                        if (self.PhotoSegmentationNameList[name] == currentRow):
                            Index = name
                    if (Index != -1):
                        if not os.path.exists(Folder_Path + "\\PhotoSeg"):
                            os.makedirs(Folder_Path + "\\PhotoSeg")
                        plt.imsave(Folder_Path + "\\PhotoSeg" + '\\' + selected_text + '.png', self.PhotoSegmentationList[Index])

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Func()
    ui.setupUiFunc(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
