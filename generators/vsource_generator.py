import random
from typing import TypedDict
from PIL.Image import Image as PImage
from PIL.ImageDraw import ImageDraw as PImageDraw
from PIL import Image, ImageDraw
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from models.generator import Generator
from models.keypoints import Keypoints
from utils.append_img_kpts import append_img_kpts
from utils.get_error import get_error
from utils.plot_image import plot_image


class VSourceDrawPoints(TypedDict):
    p0: npt.NDArray[np.int_]
    p1: npt.NDArray[np.int_]
    p2: npt.NDArray[np.int_]
    p2_0: npt.NDArray[np.int_]
    p2_1: npt.NDArray[np.int_]
    p3: npt.NDArray[np.int_]
    p4: npt.NDArray[np.int_]
    p5: npt.NDArray[np.int_]
    p6: npt.NDArray[np.int_]


class VSourceDrawMagnitudes(TypedDict):
    r2: int


class VSourceDrawParams(TypedDict):
    size: tuple[int, int]
    points: VSourceDrawPoints
    magnitudes: VSourceDrawMagnitudes


class VSourceGenerator(Generator):
    def _get_draw_params(self) -> VSourceDrawParams:
        D = self.step

        d01 = 4*D
        d12 = 3*D
        r2 = d12
        d23 = 2*D
        d24 = d23
        d15 = 2*d12
        d56 = d01

        p0 = np.array([0, 0], dtype=np.int_)
        p1 = p0 + [0, -d01]
        p2 = p1 + [0, -d12]
        p3 = p2 + [0, -d23]
        p4 = p2 + [0, d24]
        p5 = p1 + [0, -d15]
        p6 = p5 + [0, -d56]

        e_p0 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p1 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p2 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_r2 = int(r2 * (0.1 + get_error(0.1)))
        e_p3 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p4 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p5 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_p6 = [int(D*get_error(0.1)), int(D*get_error(0.1))]

        p0 += e_p0
        p1 += e_p1
        p2 += e_p2
        p3 += e_p3
        p4 += e_p4
        p5 += e_p5
        p6 += e_p6
        r2 += e_r2

        p2_0 = p2 - [r2, r2]
        p2_1 = p2 + [r2, r2]

        points: VSourceDrawPoints = {
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'p2_0': p2_0,
            'p2_1': p2_1,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6,
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

        params: VSourceDrawParams = {
            'size': (W, H),
            'points': points,
            'magnitudes': {
                'r2': r2
            }
        }

        return params

    def _draw_positive(self, draw: PImageDraw, coord: npt.NDArray[np.int_]) -> None:
        D = self.step

        p_top = coord - [0, D]
        p_bottom = coord + [0, D]
        p_left = coord - [D, 0]
        p_right = coord + [D, 0]

        e_p_top = [int(D*get_error(0.2)), int(D*get_error(0.1))]
        e_p_bottom = [-e_p_top[0], int(D*get_error(0.1))]
        e_p_left = [int(D*get_error(0.1)), int(D*get_error(0.2))]
        e_p_right = [int(D*get_error(0.1)), -e_p_left[1]]

        p_top += e_p_top
        p_bottom += e_p_bottom
        p_left += e_p_left
        p_right += e_p_right

        p_top = tuple(p_top)
        p_bottom = tuple(p_bottom)
        p_left = tuple(p_left)
        p_right = tuple(p_right)

        draw.line([p_top, p_bottom], fill=(255, 255, 255), width=self.draw_width)
        draw.line([p_left, p_right], fill=(255, 255, 255), width=self.draw_width)

    def _draw_negative(self, draw: PImageDraw, coord: npt.NDArray[np.int_]) -> None:
        D = self.step

        p_left = coord - [D, 0]
        p_right = coord + [D, 0]

        e_p_left = [int(D*get_error(0.1)), int(D*get_error(0.2))]
        e_p_right = [int(D*get_error(0.1)), -e_p_left[1]]

        p_left += e_p_left
        p_right += e_p_right

        p_left = tuple(p_left)
        p_right = tuple(p_right)

        draw.line([p_left, p_right], fill=(255, 255, 255), width=self.draw_width)

    def _draw_image(self, params: VSourceDrawParams) -> PImage:
        img = Image.new(mode='RGB', size=params['size'])

        draw = ImageDraw.Draw(img)

        p = params['points']

        draw.line([tuple(p['p0']), tuple(p['p1'])],
                  fill=(255, 255, 255), width=self.draw_width)

        draw.ellipse([tuple(p['p2_0']), tuple(p['p2_1'])],
                     fill=None, outline=(255, 255, 255), width=self.draw_width)

        draw.line([tuple(p['p5']), tuple(p['p6'])],
                  fill=(255, 255, 255), width=self.draw_width)

        self._draw_positive(draw, p['p3'])
        self._draw_negative(draw, p['p4'])

        return img

    def _get_keypoints(self, params: VSourceDrawParams):
        p = params['points']
        m = params['magnitudes']

        positive = p['p2'] - [0, m['r2']]
        negative = p['p2'] + [0, m['r2']]

        kpts = {
            'positive': positive,
            'negative': negative
        }

        return kpts


if __name__ == '__main__':
    vsource_generator = VSourceGenerator()

    img, kpts = vsource_generator.generate()

    img_kpts = append_img_kpts(img, kpts)

    plt.imshow(img_kpts)
    plt.show()
