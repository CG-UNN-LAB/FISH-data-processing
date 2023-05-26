import os

import czifile
import numpy as np
from PIL import Image


def read_czi_file_as_pil_image(filename):
    with czifile.CziFile(os.path.join(filename)) as czi:
        # Получить все каналы в файле:
        channels = czi.asarray()[0, 0, :, :, :]
        # Объедините все каналы в единое RGB-изображение:
        image_array = np.stack([channels[1], channels[2], channels[0]], axis=-1)
        # Преобразуем изображение в Pillow:
        image_array = np.squeeze(image_array)
        image = np.uint8(image_array)
        result = Image.fromarray(image)
    return result
