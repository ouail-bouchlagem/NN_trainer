import random
from time import sleep

import numpy as np


sleep_time = 0


class Image:
    def __init__(self, size=28, brush_size=None):
        self.brush_size = brush_size if brush_size else int(size * 0.05)
        self.size = size
        self.pixels = [[0 for _ in range(size)] for _ in range(size)]

    def draw_x(self):
        for _ in self.iter_draw_x():
            sleep(sleep_time)

    def iter_draw_x(self):
        top, left, bottom, right = self.top(), self.left(), self.bottom(), self.right()
        current_point = bottom, left
        while current_point[0] > top and current_point[1] < right:
            self.draw_point(*current_point)
            yield
            direction = (
                (top - current_point[0]),
                (right - current_point[1]),
            )
            angle = random.uniform(-30, 30) * (3.14 / 180)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            direction = (
                direction[0] * cos_angle - direction[1] * sin_angle,
                direction[0] * sin_angle + direction[1] * cos_angle,
            )
            length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
            if length != 0:
                direction = (direction[0] / length, direction[1] / length)

            current_point = (
                round(current_point[0] + direction[0]),
                round(current_point[1] + direction[1]),
            )

        current_point = top, left
        while current_point[0] < bottom and current_point[1] < right:
            self.draw_point(*current_point)
            yield
            direction = (
                (bottom - current_point[0]),
                (right - current_point[1]),
            )
            length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
            if length != 0:
                direction = (direction[0] / length, direction[1] / length)

            angle = random.uniform(-30, 30) * (3.14 / 180)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            direction = (
                direction[0] * cos_angle - direction[1] * sin_angle,
                direction[0] * sin_angle + direction[1] * cos_angle,
            )
            current_point = (
                round(current_point[0] + direction[0]),
                round(current_point[1] + direction[1]),
            )

    def derivative_of_circle(self, x):
        if abs(x) >= 1:
            return 0
        return -x / (1 - x**2) ** 0.5

    def draw_o(self):
        for _ in self.iter_draw_o():
            sleep(sleep_time)

    def iter_draw_o(self):
        top, left, bottom, right = self.top(), self.left(), self.bottom(), self.right()
        center = (round((top + bottom) / 2), round((left + right) / 2))

        radius = min((bottom - top), (right - left)) / 2
        noise = random.uniform(-1, 1) * self.brush_size
        circumference = 2 * 3.14 * radius
        step_size = int((self.brush_size / circumference * 360) / 2)
        for angle in range(0, 360, step_size):
            rad = angle * (3.14 / 180)
            x = center[0] + radius * np.cos(rad) + noise
            y = center[1] + radius * np.sin(rad) + noise
            self.draw_point(round(x), round(y))
            yield
            noise += random.uniform(-0.02 * self.size, 0.02 * self.size)
            if noise > self.brush_size:
                noise = self.brush_size
            if noise < -self.brush_size:
                noise = -self.brush_size

    def get_drawn_pixels_count(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.pixels[i][j] > 0:
                    count += 1
        return count

    def top(self):
        return random.randint(0, round(self.size * 0.3))

    def bottom(self):
        return random.randint(round(self.size * 0.7), self.size - 1)

    def left(self):
        return self.top()

    def right(self):
        return self.bottom()

    def draw_point(self, x, y):
        for i in range(-self.brush_size, self.brush_size + 1):
            for j in range(-self.brush_size, self.brush_size + 1):
                if 0 <= x + i < self.size and 0 <= y + j < self.size:
                    distance = (i**2 + j**2) ** 0.5
                    ratio = self.brush_size / distance if distance != 0 else 1
                    ratio = ratio * random.uniform(0.5, 1)
                    self.pixels[x + i][y + j] = max(
                        self.pixels[x + i][y + j], 255 * min(1, ratio)
                    )
