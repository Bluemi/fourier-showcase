import pygame as pg
import numpy as np
import scipy.fft


IMAGE_SIZE = 100
IMAGE_SHAPE = (IMAGE_SIZE,) * 2

RENDER_SIZE = 600
BORDER = 10
IMAGE_RECT = pg.Rect(BORDER, BORDER, RENDER_SIZE, RENDER_SIZE)
FREQUENCY_IMAGE_RECT1 = IMAGE_RECT.move(RENDER_SIZE + BORDER, 0)
FREQUENCY_IMAGE_RECT2 = FREQUENCY_IMAGE_RECT1.move(RENDER_SIZE + BORDER, 0)

DEFAULT_SCREEN_SIZE = np.array([BORDER + (RENDER_SIZE + BORDER)*3 + BORDER, RENDER_SIZE+BORDER*2])


def gray(b):
    return pg.Color(b, b, b)


class Main:
    def __init__(self):
        self.running = True
        self.screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
        self.space_image = np.zeros(IMAGE_SHAPE)
        self.frequency_space = np.zeros(IMAGE_SHAPE, dtype=complex)
        self.update_needed = True

        self.drawing = False
        self.last_flipped_index = None

    def run(self):
        while self.running:
            events = [pg.event.wait()]
            for event in events + pg.event.get():
                self.handle_event(event)
            if self.update_needed:
                self.render()
                self.update_needed = False

    def render(self):
        self.screen.fill(gray(127))

        image = image_from_np2d(self.space_image.real, RENDER_SIZE, normalize=True)
        self.screen.blit(image, IMAGE_RECT)

        frequency_image1 = image_from_np2d(self.frequency_space.real, RENDER_SIZE)
        self.screen.blit(frequency_image1, FREQUENCY_IMAGE_RECT1)

        frequency_image2 = image_from_np2d(self.frequency_space.imag, RENDER_SIZE)
        self.screen.blit(frequency_image2, FREQUENCY_IMAGE_RECT2)

        print(np.max(self.frequency_space))

        pg.display.update()

    def handle_event(self, event):
        if event.type == pg.QUIT:
            self.running = False
        if event.type == pg.MOUSEBUTTONDOWN:
            # print(self.frequency_space.flags)
            # print(np.imag(self.frequency_space).copy().flags)
            self.flip_point(event.pos)
            self.drawing = True
        if event.type == pg.MOUSEBUTTONUP:
            self.drawing = False
            self.last_flipped_index = None
        if event.type == pg.MOUSEMOTION:
            if self.drawing:
                self.flip_point(event.pos)
        if event.type == pg.KEYDOWN:
            if event.unicode == 's':
                frequency = 2
                self.space_image = np.cos(
                    np.linspace(0, 2*np.pi * frequency, IMAGE_SIZE)
                    .repeat(IMAGE_SIZE)
                    .reshape(IMAGE_SIZE, IMAGE_SIZE)
                )
                self.update_frequencies()
                self.update_needed = True
        else:
            # print(event)
            pass

    def flip_point(self, pos):
        rects = [IMAGE_RECT, FREQUENCY_IMAGE_RECT1, FREQUENCY_IMAGE_RECT2]
        rect_index = -1
        rect = None
        for i, r in enumerate(rects):
            if r.collidepoint(pos):
                rect_index = i
                rect = r
                break
        if rect is None:
            return
        rect_top_left = np.array(rect.topleft)
        mouse_pos = np.array(pos)
        index = ((mouse_pos - rect_top_left) * (IMAGE_SIZE / RENDER_SIZE)).astype(int)
        index = tuple(np.minimum(np.maximum(index, 0), IMAGE_SIZE - 1))

        flipped_index = (*index, rect_index)
        if self.last_flipped_index == flipped_index:
            return
        self.last_flipped_index = flipped_index

        # mutate image
        if rect_index == 0:
            self.space_image[index] = 1 - self.space_image[index]
        elif rect_index == 1:
            self.frequency_space.real[index] = 1 - self.frequency_space.real[index]
        elif rect_index == 2:
            imag_cpy = np.imag(self.frequency_space).copy()
            imag_cpy[index] = 1 - imag_cpy[index]
            self.frequency_space.imag = imag_cpy
        if rect_index == 0:
            self.update_frequencies()
        elif rect_index in (1, 2):
            self.update_space()
        self.update_needed = True

    def update_space(self):
        self.space_image = scipy.fft.ifft2(self.frequency_space)

    def update_frequencies(self):
        self.frequency_space = scipy.fft.fft2(self.space_image)


def image_from_np2d(a, scale_shape, normalize=False):
    if isinstance(scale_shape, int):
        scale_shape = (scale_shape,) * 2
    a = a.real
    mini = np.min(a)
    if mini < 0:
        a = a - np.min(a)
    maxi = np.max(a)
    if maxi > 1 or (normalize and maxi != 0):
        a /= maxi
    a = a * 255
    shape = a.shape
    a = np.repeat(a, 3, axis=-1).reshape(*shape, 3)
    image = pg.surfarray.make_surface(a.astype(int))
    return pg.transform.scale(image, scale_shape)


def main():
    pg.init()
    main_instance = Main()
    main_instance.run()


if __name__ == '__main__':
    main()
