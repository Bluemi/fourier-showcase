import pygame as pg
import numpy as np
import scipy.fft

from image_loader import ImageLoader

RENDER_SCALE_1D = 10

IMAGE_SIZE = 100
IMAGE_SHAPE = (IMAGE_SIZE,) * 2

RENDER_SIZE = 600
BORDER = 10
IMAGE_RECT = pg.Rect(BORDER, BORDER, RENDER_SIZE, RENDER_SIZE)
FREQUENCY_IMAGE_RECT1 = IMAGE_RECT.move(RENDER_SIZE + BORDER, 0)
FREQUENCY_IMAGE_RECT2 = FREQUENCY_IMAGE_RECT1.move(RENDER_SIZE + BORDER, 0)

DEFAULT_SCREEN_SIZE = np.array([BORDER + (RENDER_SIZE + BORDER)*2, RENDER_SIZE+BORDER*2])

TRANSFORMS = ['fft', 'dct']


def gray(b):
    return pg.Color(b, b, b)


class Main:
    def __init__(self):
        self.running = True
        self.mode = '1'

        # mode 1
        self.samples = np.zeros((10,), dtype=complex)
        self.spectrum = np.fft.fft(self.samples)
        self.show_imaginary = False

        self.dragging = False
        self.last_mouse_button = None

        # mode 2
        self.transform_index = 0
        self.screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
        self.space_image = np.zeros(IMAGE_SHAPE)
        self.frequency_space = np.zeros(IMAGE_SHAPE, dtype=complex)
        self.update_frequencies()
        self.update_needed = True

        self.frequency_vert = 0
        self.frequency_hori = 0

        self.drawing = False
        self.last_flipped_index = None
        self.image_loader = ImageLoader(IMAGE_SHAPE)

    def run(self):
        while self.running:
            events = [pg.event.wait()]
            for event in events + pg.event.get():
                self.handle_event(event)
            if self.update_needed:
                self.render()
                self.update_needed = False

    def render(self):
        self.screen.fill(gray(80))

        if self.mode == '1':
            self.render_1d()
        elif self.mode == '2':
            image = image_from_np2d(self.space_image.real, RENDER_SIZE, normalize=True)
            self.screen.blit(image, IMAGE_RECT)

            frequency_image1 = image_from_np2d(self.frequency_space.real, RENDER_SIZE)
            self.screen.blit(frequency_image1, FREQUENCY_IMAGE_RECT1)

            frequency_image2 = image_from_np2d(self.frequency_space.imag, RENDER_SIZE)
            self.screen.blit(frequency_image2, FREQUENCY_IMAGE_RECT2)

        pg.display.update()

    def render_1d(self):
        # colors
        background_color = gray(40)
        hline_color = gray(60)
        real_color = pg.Color(40, 120, 250)
        imag_color = pg.Color(210, 80, 20)

        # render samples
        pg.draw.rect(self.screen, background_color, IMAGE_RECT)  # draw background
        y_line = BORDER + RENDER_SIZE / 2  # y-pos of h-line
        pg.draw.line(self.screen, hline_color, (BORDER, y_line), (BORDER+RENDER_SIZE, y_line))
        if self.show_imaginary:
            self.draw_1d_samples(self.samples.imag, BORDER, imag_color)
        self.draw_1d_samples(self.samples.real, BORDER, real_color)

        # render real part
        left = FREQUENCY_IMAGE_RECT1.left
        pg.draw.rect(self.screen, background_color, FREQUENCY_IMAGE_RECT1)  # draw background
        pg.draw.line(self.screen, hline_color, (left, y_line), (left+RENDER_SIZE, y_line))
        self.draw_1d_samples(self.spectrum.imag, left, imag_color)
        self.draw_1d_samples(self.spectrum.real, left, real_color)

    def draw_1d_samples(self, samples, left, color):
        y_line = BORDER + RENDER_SIZE / 2  # y-pos of h-line
        y_positions = y_line - (samples * RENDER_SCALE_1D)  # y-positions of samples
        n_samples = len(samples)
        sample_width = int((RENDER_SIZE / n_samples) * 0.3)
        x_positions = (np.linspace(0, RENDER_SIZE, n_samples,
                                   endpoint=False) + left + RENDER_SIZE / n_samples / 2).round().astype(int)
        for x, y in zip(x_positions, y_positions):
            pg.draw.line(self.screen, color, (x - sample_width, y), (x + sample_width, y), width=3)
            pg.draw.line(self.screen, color, (x, y_line), (x, y), width=3)

    def handle_mouse_1d(self, pos, rect, samples_to_update):
        if rect.collidepoint(pos):
            x_pos = pos[0] - rect.left
            n_samples = len(self.samples)
            x_positions = (np.linspace(0, RENDER_SIZE, n_samples, endpoint=False) + RENDER_SIZE / n_samples / 2).round().astype(int)
            x_index = np.argmin(np.abs(x_positions - x_pos))

            y_pos = ((BORDER + RENDER_SIZE / 2) - pos[1]) / RENDER_SCALE_1D
            samples_to_update[x_index] = y_pos

            return True
        return False

    def handle_all_mouse_1d(self, pos, button):
        if button == 1:
            samples_to_modify = self.samples.real
        elif button == 3:
            samples_to_modify = self.samples.imag
        else:
            return
        if self.handle_mouse_1d(pos, IMAGE_RECT, samples_to_modify):
            self.update_frequencies_from_samples_1d()
            self.update_needed = True

        if button == 1:
            spectrum_to_modify = self.spectrum.real
        elif button == 3:
            spectrum_to_modify = self.spectrum.imag
        else:
            return
        if self.handle_mouse_1d(pos, FREQUENCY_IMAGE_RECT1, spectrum_to_modify):
            self.update_samples_from_frequencies_1d()
            self.update_needed = True

    def update_frequencies_from_samples_1d(self):
        self.spectrum = np.fft.fft(self.samples)

    def update_samples_from_frequencies_1d(self):
        self.samples = np.fft.ifft(self.spectrum)

    def handle_event(self, event):
        if event.type == pg.QUIT:
            self.running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_1:
                self.mode = '1'
                self.update_needed = True
            elif event.key == pg.K_2:
                self.mode = '2'
                self.update_needed = True
        if self.mode == '1':
            if event.type == pg.MOUSEBUTTONDOWN:
                self.dragging = True
                self.last_mouse_button = event.button
                self.handle_all_mouse_1d(event.pos, event.button)
            elif event.type == pg.MOUSEBUTTONUP:
                self.dragging = False
                self.last_mouse_button = None
            elif event.type == pg.MOUSEMOTION:
                if self.dragging:
                    self.handle_all_mouse_1d(event.pos, self.last_mouse_button)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_PLUS:
                    samples_to_add = int(round(len(self.samples) * 0.2))
                    samples_to_add = max(samples_to_add, 1)
                    new_samples = np.zeros(samples_to_add, dtype=complex)
                    self.samples = np.concatenate([self.samples, new_samples])
                    self.update_frequencies_from_samples_1d()
                    self.update_needed = True
                elif event.key == pg.K_MINUS:
                    new_size = len(self.samples) * (1 / 1.2)
                    new_size = int(max(min(new_size, len(self.samples)-1), 1))
                    self.samples = self.samples[:new_size]
                    self.update_frequencies_from_samples_1d()
                    self.update_needed = True
                elif event.key == pg.K_i:
                    self.show_imaginary = not self.show_imaginary
                    self.update_needed = True
                elif event.key == pg.K_0:
                    self.samples = np.zeros((len(self.samples),), dtype=complex)
                    self.update_frequencies_from_samples_1d()
                    self.update_needed = True
        elif self.mode == '2':
            if event.type == pg.MOUSEBUTTONDOWN:
                self.flip_point(event.pos)
                self.drawing = True
            if event.type == pg.MOUSEBUTTONUP:
                self.drawing = False
                self.last_flipped_index = None
            if event.type == pg.MOUSEMOTION:
                if self.drawing:
                    self.flip_point(event.pos, False)
            if event.type == pg.KEYDOWN:
                if event.unicode == 's':
                    self.frequency_vert += 1
                    print('freq_vert: ', self.frequency_vert, 'freq_hori', self.frequency_hori)
                    self.generate_image_with_frequency()
                elif event.unicode == 'S':
                    self.frequency_vert -= 1
                    print('freq_vert: ', self.frequency_vert, 'freq_hori', self.frequency_hori)
                    self.generate_image_with_frequency()
                if event.unicode == 'd':
                    self.frequency_hori += 1
                    print('freq_vert: ', self.frequency_vert, 'freq_hori', self.frequency_hori)
                    self.generate_image_with_frequency()
                elif event.unicode == 'D':
                    self.frequency_hori -= 1
                    print('freq_vert: ', self.frequency_vert, 'freq_hori', self.frequency_hori)
                    self.generate_image_with_frequency()
                elif event.unicode == 'i':
                    self.space_image = self.image_loader.next_image()
                    self.update_frequencies()
                    self.update_needed = True
                elif event.unicode == 'I':
                    self.space_image = self.image_loader.prev_image()
                    self.update_frequencies()
                    self.update_needed = True
                elif event.unicode == 'n':
                    self.transform_index = (self.transform_index + 1) % len(TRANSFORMS)
                    self.update_frequencies()
                    self.update_needed = True
                    print(f'Using {TRANSFORMS[self.transform_index]}')
                elif event.key == 27:
                    self.running = False
            else:
                self.update_needed = True
        elif self.mode == '1':
            pass

    def generate_image_with_frequency(self):
        lin_space = np.linspace(0, 2 * np.pi, IMAGE_SIZE)\
            .repeat(IMAGE_SIZE)\
            .reshape(IMAGE_SIZE, IMAGE_SIZE)
        space_image = np.sin(lin_space * self.frequency_vert) + np.sin(lin_space.T * self.frequency_hori)

        self.space_image = (space_image + 1) / 2
        self.update_frequencies()
        self.update_needed = True

    def flip_point(self, pos, choose_value=True):
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
            if choose_value and abs(self.space_image[index]) < 0.00001:
                self.space_image[index] = 1
            else:
                self.space_image[index] = 0
        elif rect_index in (1, 2):
            if choose_value and abs(self.frequency_space[index]) < 0.00001:
                if self.transform_index == 0:
                    self.frequency_space[index] = IMAGE_SIZE**2 + IMAGE_SIZE**2*1j
                elif self.transform_index == 1:
                    self.frequency_space[index] = IMAGE_SIZE ** 2
                else:
                    raise ValueError('Unknown transform with index: ', self.transform_index)
            else:
                self.frequency_space[index] = 0

        if rect_index == 0:
            self.update_frequencies()
        elif rect_index in (1, 2):
            self.update_space()
        self.update_needed = True

    def update_space(self):
        if self.transform_index == 0:
            space_image = scipy.fft.ifft2(self.frequency_space)
        elif self.transform_index == 1:
            space_image = scipy.fft.idct(self.frequency_space)
        else:
            raise ValueError('Unknown transform with index: ', self.transform_index)
        self.space_image = (space_image + 1) / 2

    def update_frequencies(self):
        space_image = self.space_image * 2 - 1
        if self.transform_index == 0:
            self.frequency_space = scipy.fft.fft2(space_image)
        elif self.transform_index == 1:
            self.frequency_space = scipy.fft.dct(space_image)
        else:
            raise ValueError('Unknown transform with index: ', self.transform_index)


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
