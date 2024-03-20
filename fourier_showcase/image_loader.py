import os

import numpy as np
from PIL import Image

ALLOWED_FILE_TYPES = {'.png', '.jpg'}


class ImageLoader:
    def __init__(self, image_size):
        self.image_index = 0
        if os.path.isdir('images'):
            self.image_paths = [p for p in os.listdir('images') if os.path.splitext(p)[1] in ALLOWED_FILE_TYPES]
        else:
            self.image_paths = []
        self.image_size = image_size
        if self.image_paths:
            self.image = self.load_image()
        else:
            self.image = None

    def load_image(self):
        path = os.path.join('images', self.image_paths[self.image_index])
        image = Image.open(path).resize(self.image_size)
        # noinspection PyTypeChecker
        image = np.array(image)[..., :3]
        image = np.moveaxis(image, [0, 1], [1, 0])
        # grayscale
        if len(image.shape) == 3:
            image = (image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114) / 255
        return image

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self.image = self.load_image()
        return self.image

    def prev_image(self):
        self.image_index -= 1
        if self.image_index < 0:
            self.image_index = len(self.image_paths) - 1
        self.image = self.load_image()
        return self.image
