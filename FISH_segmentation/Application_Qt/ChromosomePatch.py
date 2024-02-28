import enum
import os
import csv
import cv2
import imutils
import numpy as np
import scipy
import skimage
from matplotlib import pyplot as plt
from ultralytics import YOLO
from scipy import ndimage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cell:
    @enum.unique
    class CellType(enum.Enum):
        EXPLODED = 0
        WHOLE = 1

    def __init__(self, masked_area: np.ma.MaskedArray, cell_type: CellType):
        self.masked_area = masked_area
        self.cell_type = cell_type

        self.red_chromosomes = []
        self.green_chromosomes = []
        self.center_of_mass = []
        self.Type = -1

    def add_center_of_mass(self, center_of_mass):
        self.center_of_mass.append(center_of_mass)

    def add_red_chromosome(self, red_chromosome):
        self.red_chromosomes.append(red_chromosome)

    def add_green_chromosome(self, green_chromosome):
        self.green_chromosomes.append(green_chromosome)


class ChromosomeCellDetector:
    RedChromosome = 0
    GreenChromosome = 0
    NumberExplode = 0
    NumberWhole = 0
    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..\\Model\\my_yolov8_model_core_segmentation_plus_plus.pt")
    CELLS_DETECTOR = YOLO(MODEL_PATH)

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image
        self.cells: list[Cell] = []
        self.Radius = []

    def plot(self, ax=None):
        ax.imshow(self.image)
        for cell in self.cells:
            mask = np.invert(cell.masked_area.mask[..., 0]).astype(np.uint8)
            contour_color = 'green' if cell.cell_type == Cell.CellType.WHOLE else 'red'
            ax.contour(mask, colors=contour_color, linewidths=0.5, alpha=0.5)

            for p in cell.red_chromosomes:
                circle = plt.Circle((p[1], p[0]), radius=3, color='red', fill=False, linestyle='--')
                ax.add_patch(circle)

            for p in cell.green_chromosomes:
                circle = plt.Circle((p[1], p[0]), radius=3, color='green', fill=False, linestyle='--')
                ax.add_patch(circle)
        return ax

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape
        if ch == 3:
            return rgba
        assert ch == 4, 'RGBA image has 4 channels.'
        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        a = np.asarray(a, dtype='float32') / 255.0
        R, G, B = background
        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B
        return np.asarray(rgb, dtype='uint8')

    def find_cells(self, confidence: float = 0.5):
        if self.cells:
            self.cells.clear()

        predictions = ChromosomeCellDetector.CELLS_DETECTOR.predict(
            self.image.copy(),
            classes=[0, 1],
            conf=confidence,
        )
        self.Radius.clear()
        for prediction in predictions:
            masks = prediction.masks.data.numpy().transpose(1, 2, 0)
            classes = prediction.boxes.cls.data.numpy()

            for mask, cls in zip(np.rollaxis(masks, 2), classes):
                mask = cv2.resize(mask, dsize=self.image.shape[:2], interpolation=cv2.INTER_LINEAR)
                mask3d = (np.repeat(mask[..., np.newaxis], 3, axis=-1) > 0).astype(bool)

                # TODO: opencv and skimage give slightly different resized mask
                # mask = skimage.transform.resize(
                #     mask.astype(bool),
                #     output_shape=self.image.shape[:2],
                #     order=0,
                #     preserve_range=True,
                #     anti_aliasing=False
                # )
                # mask3d = np.repeat(mask[..., np.newaxis], 3, axis=-1)

                masked_image = np.ma.masked_where(np.invert(mask3d), self.image)

                cell = Cell(masked_image, Cell.CellType(int(cls)))
                self.cells.append(cell)

                # Найдем координаты центра масс каждой клетки
                if cell.cell_type == Cell.CellType.EXPLODED or cell.cell_type == Cell.CellType.WHOLE:
                    labeled_mask, num_labels = ndimage.label(mask)
                    for label in range(1, num_labels + 1):
                        np.argwhere(labeled_mask == label)
                        center_of_mass = ndimage.center_of_mass(mask, labeled_mask, label)
                        cell.add_center_of_mass(center_of_mass)

                        distances = np.sqrt(np.sum((np.argwhere(labeled_mask == label) 
                                                    - np.array(center_of_mass))**2, axis=1))
                        # Находим максимальное расстояние, которое и будет радиусом вокруг центра масс
                        self.Radius.append(np.max(distances))

        # Доп.Информация:
        names = ChromosomeCellDetector.CELLS_DETECTOR.names
        self.NumberWhole = 0
        self.NumberExplode = 0
        for r in predictions:
            for c in r.boxes.cls:
                if names[int(c)] == "Whole cell":
                    self.NumberWhole += 1
                if names[int(c)] == "Explode cell":
                    self.NumberExplode += 1
        print("Explode:")
        print(self.NumberExplode)
        print("Whole:")
        print(self.NumberWhole)
        return self.NumberExplode, self.NumberWhole

    def write_to_csv(self, output_file, folder_path, file_name):
        file_exists = os.path.isfile(output_file)
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')

            # Если файл только что создан, добавляем заголовок
            if not file_exists or csvfile.tell() == 0:
                header = ["Folder Path", "File Name", "Cell Number", "Center X", "Center Y",
                          "Green Chromosomes", "Red Chromosomes", "Cell Type"]
                writer.writerow(header)

            if not file_exists:
                logger.info(f" Файл {output_file} был создан, заголовок добавлен.")
            else:
                logger.info(f" Файл {output_file} уже существует, данные будут записаны к уже существующим.")

            # Далее ваш код записи данных в CSV, например:
            for idx, cell in enumerate(self.cells):
                for center_of_mass in cell.center_of_mass:
                    row_data = [
                        folder_path,
                        file_name,
                        idx + 1,
                        center_of_mass[1],
                        center_of_mass[0],
                        len(cell.green_chromosomes),
                        len(cell.red_chromosomes),
                        "Exploded" if cell.cell_type == Cell.CellType.EXPLODED else "Whole",
                    ]
                    if (cell.cell_type == Cell.CellType.EXPLODED):
                        cell.Type = 0
                    else:
                        cell.Type = 1
                    writer.writerow(row_data)

    def detect_chromosomes(self):
        unsharped_image = ChromosomeCellDetector.__unsharp_mask(self.image,
                                                                kernel_size=(5, 5),
                                                                sigma=5.0,
                                                                amount=5.0,
                                                                threshold=100)
        red_channel, green_channel = unsharped_image[..., 0], unsharped_image[..., 1]

        red_chromosome_candidates = ChromosomeCellDetector.__get_chromosome_candidates(red_channel)
        green_chromosome_candidates = ChromosomeCellDetector.__get_chromosome_candidates(green_channel)

        closeness = 1.0
        self.RedChromosome = 0
        self.GreenChromosome = 0
        self.__filter_chromosomes(
            red_chromosome_candidates,
            'red',
            closeness=closeness)
        self.__filter_chromosomes(
            green_chromosome_candidates,
            'green',
            closeness=closeness)

    def __filter_chromosomes(
            self,
            chromosome_candidates: np.ndarray,
            chromosome_type: str,
            closeness: float = 1.0):
        accepted = np.zeros(chromosome_candidates.shape[0], dtype=bool)

        for idx, candidate in enumerate(chromosome_candidates):
            for cell in self.cells:
                mask = np.invert(cell.masked_area.mask[..., 0]).astype(np.uint8)

                # TODO: to fix, sometimes opencv on certain images crashes (IDGAF why)
                contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contour = imutils.grab_contours(contour)[0]

                distance = cv2.pointPolygonTest(contour, candidate[::-1], measureDist=True)
                inside_cell = distance > 0.0
                almost_on_cell_border = distance <= closeness

                if not accepted[idx] and inside_cell and not almost_on_cell_border:
                    accepted[idx] = True
                    if chromosome_type == 'red':
                        cell.add_red_chromosome(candidate)
                        self.RedChromosome += 1
                    elif chromosome_type == 'green':
                        cell.add_green_chromosome(candidate)
                        self.GreenChromosome += 1
                    break

    @staticmethod
    def __unsharp_mask(
            image: np.ndarray,
            kernel_size: tuple[int, int] = (5, 5),
            sigma: float = 0.0,
            amount: float = 1.0,
            threshold: int = 0):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)

        return sharpened

    @staticmethod
    def __get_chromosome_candidates(image: np.ndarray):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_type = cv2.MORPH_GRADIENT

        _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        morph = cv2.morphologyEx(thresh, morph_type, kernel)
        labels, num_labels = skimage.measure.label(morph, background=0, return_num=True, connectivity=1)
        points = scipy.ndimage.center_of_mass(labels, labels, range(1, num_labels + 1))

        return np.array(points)
