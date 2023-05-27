import czifile
import numpy as np


def read_czi_image(filename, norm=True):
    with czifile.CziFile(filename) as czi:
        image = czi.asarray().squeeze()
        # image = np.moveaxis(image, 0, -1)[..., ::-1]  # swap channels to order -> RGB
        image = np.stack([image[1], image[2], image[0]], axis=-1)  # swap channels to order -> RGB

        if norm:
            info = np.iinfo(image.dtype)
            image = image.astype(np.float64) / info.max  # normalize the data to 0-1
            image = (255 * image).astype(np.uint8)  # normalize the data to 0-255

        image = np.ascontiguousarray(image)
        metadata = czi.metadata(raw=True)

        return image, metadata
