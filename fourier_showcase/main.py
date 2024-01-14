#!/usr/bin/env python3

import sys
import pygame as pg
import numpy as np

RENDER_SCALE_1D = 10

IMAGE_SIZE = 100
IMAGE_SHAPE = (IMAGE_SIZE,) * 2

RENDER_SIZE = 600
BORDER = 10
IMAGE_RECT = pg.Rect(BORDER, BORDER, RENDER_SIZE, RENDER_SIZE)
FREQUENCY_IMAGE_RECT1 = IMAGE_RECT.move(RENDER_SIZE + BORDER, 0)
FREQUENCY_IMAGE_RECT2 = FREQUENCY_IMAGE_RECT1.move(RENDER_SIZE + BORDER, 0)

DEFAULT_SCREEN_SIZE_1D = np.array([BORDER + (RENDER_SIZE + BORDER) * 2, RENDER_SIZE + BORDER * 2])


def gray(b):
    return pg.Color(b, b, b)


class Main:
    def __init__(self):
        self.samples = np.zeros((10,), dtype=complex)
        self.spectrum = np.fft.fft(self.samples)

        self.show_imaginary = False
        self.frequency = 0

        self.dragging = False
        self.last_mouse_button = None

        self.screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE_1D)
        self.update_needed = True
        self.running = True

    def run(self):
        while self.running:
            events = [pg.event.wait()]
            events = events + pg.event.get()
            self.handle_events(events)

    def handle_events(self, events):
        for event in events:
            self.handle_event(event)
        if self.update_needed:
            self.render()
            self.update_needed = False

    def render(self):
        self.screen.fill(gray(80))
        self.render_1d()
        pg.display.flip()

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
            x_positions = (
                    np.linspace(0, RENDER_SIZE, n_samples, endpoint=False) + RENDER_SIZE / n_samples / 2
            ).round().astype(int)
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
        elif event.type == pg.MOUSEBUTTONDOWN:
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
            elif event.key == pg.K_c:
                self.frequency += -1 if pg.key.get_mods() & pg.KMOD_SHIFT else 1
                space = np.linspace(0, 2.0*np.pi, len(self.samples), endpoint=False, dtype=complex)
                self.samples = np.cos(space * self.frequency) * 4.0
                self.update_frequencies_from_samples_1d()
                self.update_needed = True
            elif event.key == pg.K_s:
                self.frequency += -1 if pg.key.get_mods() & pg.KMOD_SHIFT else 1
                space = np.linspace(0, 2.0 * np.pi, len(self.samples), endpoint=False, dtype=complex)
                self.samples = np.sin(space * self.frequency) * 4.0
                self.update_frequencies_from_samples_1d()
                self.update_needed = True


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
    if "pyodide" in sys.modules:
        # noinspection PyUnresolvedReferences
        pg.event.register_event_callback(main_instance.handle_events)
        return main_instance
    else:
        main_instance.run()


if __name__ == '__main__':
    main()
