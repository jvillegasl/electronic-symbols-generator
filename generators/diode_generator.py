import random
from typing import TypedDict
from PIL import Image, ImageDraw
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt

from models.generator import Generator
from utils.get_error import get_error
from utils.plot_image import plot_image


class DiodeDrawPoints(TypedDict):
    p0: npt.NDArray[np.int_]
    p1: npt.NDArray[np.int_]
    p2: npt.NDArray[np.int_]
    p3: npt.NDArray[np.int_]
    p4: npt.NDArray[np.int_]
    p5: npt.NDArray[np.int_]
    p6: npt.NDArray[np.int_]
    p7: npt.NDArray[np.int_]


class DiodeDrawParams(TypedDict):
    points: DiodeDrawPoints
    size: tuple[int, int]


class DiodeGenerator(Generator):
    """
                p2           p5
                |   -        |
                d12     -    d45
                |           -|
    p0----d01---p1....d14...p4----d47---p7
                |           -|
                d13     -    d46
                |   -        |
                p3           p6
    """

    def _get_draw_params(self) -> DiodeDrawParams:
        D = self.step

        d01 = 5*D
        d12 = 3*D
        d13 = 3*D
        d14 = 3*D
        d45 = 3*D
        d46 = 3*D
        d47 = 5*D

        p0 = np.array([0, 0], dtype=np.int_)
        p1 = p0 + [d01, 0]
        p2 = p1 + [0, d12]
        p3 = p1 - [0, d13]
        p4 = p1 + [d14, 0]
        p5 = p4 + [0, d45]
        p6 = p4 - [0, d46]
        p7 = p4 + [d47, 0]

        p0 = p0 + [int(get_error(0.5, True) * D), int(get_error(0.3) * D)]

        e = int(get_error(0.2) * D)
        p1 = p1 + [int(get_error(0.2) * D), int(get_error(0.5) * D)]
        p2 = p2 + [e, int(get_error(0.5) * D)]
        p3 = p3 + [-e, int(get_error(0.5) * D)]

        e = int(get_error(0.2) * D)
        p4 = p4 + [int(get_error(0.2) * D), int(get_error(0.5) * D)]
        p5 = p5 + [e, int(get_error(0.5) * D)]
        p6 = p6 + [-e, int(get_error(0.5) * D)]

        p7 = p7 + [int(get_error(0.5) * D), int(get_error(0.5) * D)]

        points: DiodeDrawPoints = {
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6,
            'p7': p7
        }

        y_coords = [points[key][1] for key in points]
        x_coords = [points[key][0] for key in points]

        min_y = min(y_coords)
        max_y = max(y_coords)

        min_x = min(x_coords)
        max_x = max(x_coords)

        H = max_y - min_y
        W = max_x - min_x

        padding = {
            'top': int(0.2 * H * random.uniform(0, 1)),
            'bottom': int(0.2 * H * random.uniform(0, 1)),
            'left': int(0.2 * W * random.uniform(0, 1)),
            'right': int(0.2 * W * random.uniform(0, 1))
        }

        H += padding['top'] + padding['bottom']
        W += padding['left'] + padding['right']

        for key in points:
            points[key] -= [0, min_y]
            points[key] += [padding['left'], padding['top']]

        params: DiodeDrawParams = {
            'points': points,
            'size': (W, H)
        }

        return params

    def _draw_image(self, params: DiodeDrawParams):
        points = params['points']
        size = params['size']

        img = Image.new(mode='RGB', size=size)

        draw = ImageDraw.Draw(img)
        draw.line([tuple(points['p0']), tuple(points['p1'])],
                  fill=(255, 255, 255), width=self.draw_width)

        draw.line([tuple(points['p2']), tuple(points['p3'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(points['p2']), tuple(points['p4'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(points['p3']), tuple(points['p4'])],
                  fill=(255, 255, 255), width=self.draw_width)

        draw.line([tuple(points['p5']), tuple(points['p6'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(points['p4']), tuple(points['p7'])],
                  fill=(255, 255, 255), width=self.draw_width)

        return img

    def _get_keypoints(self, params: DiodeDrawParams):
        D = self.step

        points = params['points']

        anode = points['p1'] + \
            [int(D * get_error(0.15)), int(D * get_error(0.15))]
        catode = points['p4'] + \
            [int(D * get_error(0.15)), int(D * get_error(0.15))]

        kpts = {'anode': anode, 'catode': catode}

        return kpts


if __name__ == '__main__':
    diode_generator = DiodeGenerator()

    img, kpts = diode_generator.generate()

    plot_image(img, kpts)
