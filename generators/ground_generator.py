import random
from typing import TypedDict
from PIL.Image import Image as PImage
from PIL.ImageDraw import ImageDraw as PImageDraw
from PIL import Image, ImageDraw
import numpy as np
import numpy.typing as npt

from models.generator import Generator
from models.keypoints import Keypoints
from utils.get_error import get_error
from utils.plot_image import plot_image


class GroundDrawPoints(TypedDict):
    p0: npt.NDArray[np.int_]
    p1: npt.NDArray[np.int_]
    p1_0: npt.NDArray[np.int_]
    p1_1: npt.NDArray[np.int_]
    p2: npt.NDArray[np.int_]
    p2_0: npt.NDArray[np.int_]
    p2_1: npt.NDArray[np.int_]
    p3: npt.NDArray[np.int_]
    p3_0: npt.NDArray[np.int_]
    p3_1: npt.NDArray[np.int_]


class GroundDrawMagnitudes(TypedDict):
    d1: int
    d2: int
    d3: int


class GroundDrawParams(TypedDict):
    points: GroundDrawPoints
    size: tuple[int, int]
    magnitudes: GroundDrawMagnitudes


class GroundGenerator(Generator):
    def _get_draw_params(self) -> GroundDrawParams:
        D = self.step

        d01 = 4*D
        d12 = int(1*D)
        d23 = d12
        d1 = 6*D
        d2 = 4*D
        d3 = 2*D

        p0 = np.array([0, 0], dtype=np.int_)

        p1 = p0 + [0, d01]
        p1_0 = (p1 - [d1/2, 0]).astype(int)
        p1_1 = p1_0 + [d1, 0]

        p2 = p1 + [0, d12]
        p2_0 = (p2 - [d2/2, 0]).astype(int)
        p2_1 = p2_0 + [d2, 0]

        p3 = p2 + [0, d23]
        p3_0 = (p3 - [d3/2, 0]).astype(int)
        p3_1 = p3_0 + [d3, 0]

        e_p0 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p1 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p2 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p3 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p1_0 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p1_1 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p2_0 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p2_1 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p3_0 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p3_1 = [int(D*get_error(0.1)), int(D*get_error(0.1))]

        p0 += e_p0
        p1 += e_p1
        p2 += e_p2
        p3 += e_p3
        p1_0 += e_p1_0
        p1_1 += e_p1_1
        p2_0 += e_p2_0
        p2_1 += e_p2_1
        p3_0 += e_p3_0
        p3_1 += e_p3_1

        points: GroundDrawPoints = {
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'p1_0': p1_0,
            'p1_1': p1_1,
            'p2_0': p2_0,
            'p2_1': p2_1,
            'p3_0': p3_0,
            'p3_1': p3_1,
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
            points[key] -= [min_x, min_y]
            points[key] += [padding['left'], padding['top']]

        params: GroundDrawParams = {
            'points': points,
            'size': (W, H),
            'magnitudes': {
                'd1': d1,
                'd2': d2,
                'd3': d3,
            }
        }

        return params

    def _draw_image(self, params: GroundDrawParams):
        img = Image.new(mode='RGB', size=params['size'])

        draw = ImageDraw.Draw(img)

        p = params['points']

        draw.line([tuple(p['p0']), tuple(p['p1'])],
                  fill=(255, 255, 255), width=self.draw_width)

        p1_0 = tuple(p['p1_0'])
        p1_1 = tuple(p['p1_1'])
        p2_0 = tuple(p['p2_0'])
        p2_1 = tuple(p['p2_1'])
        p3_0 = tuple(p['p3_0'])
        p3_1 = tuple(p['p3_1'])

        draw.line([p1_0, p1_1], fill=(255, 255, 255), width=self.draw_width)
        draw.line([p2_0, p2_1], fill=(255, 255, 255), width=self.draw_width)
        draw.line([p3_0, p3_1], fill=(255, 255, 255), width=self.draw_width)

        return img

    def _get_keypoints(self, params: GroundDrawParams):
        p = params['points']

        kpts = {
            'ground': p['p0']
        }

        return kpts


if __name__ == '__main__':
    generator = GroundGenerator()

    img, kpts = generator.generate()

    plot_image(img, kpts)
