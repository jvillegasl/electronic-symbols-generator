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
from utils.get_angle_uvector import get_angle_uvector
from utils.get_error import get_error
from utils.plot_image import plot_image


class NPNDrawPoints(TypedDict):
    p0: npt.NDArray[np.int_]
    p1: npt.NDArray[np.int_]
    p2: npt.NDArray[np.int_]
    p3: npt.NDArray[np.int_]
    p4: npt.NDArray[np.int_]
    p5: npt.NDArray[np.int_]
    p6: npt.NDArray[np.int_]
    p7: npt.NDArray[np.int_]
    p8: npt.NDArray[np.int_]
    p9: npt.NDArray[np.int_]


class NPNDrawParams(TypedDict):
    points: NPNDrawPoints
    size: tuple[int, int]


class NPNGenerator(Generator):
    def _get_draw_params(self) -> NPNDrawParams:
        D = self.step

        d01 = 3*D
        d23 = 5*D
        d45 = 2*D
        d56 = 2*D
        d78 = d45
        d89 = d56

        u45 = np.array([1, -1], dtype=np.int_)
        u45 = u45 / np.linalg.norm(u45)
        u78 = np.array([1, 1], dtype=np.int_)
        u78 = u78 / np.linalg.norm(u78)

        p0 = np.array([0, 0], dtype=np.int_)
        p1 = p0 + [d01, 0]
        p2 = p1 - [0, int(d23/2)]
        p3 = p2 + [0, d23]
        p4 = p1 - [0, int(d23/4)]
        p5 = (p4 + u45*d45).astype(int)
        p6 = p5 - [0, d56]
        p7 = p1 + [0, int(d23/4)]
        p8 = (p7 + u78*d78).astype(int)
        p9 = p8 + [0, d89]

        e_p0 = [int(D*get_error(0.2)), int(D*get_error(0.2))]
        e_p1 = [int(D*get_error(0.2)), -e_p0[1]]
        e_p2 = [int(D*get_error(0.2)), int(D*get_error(0.2))]
        e_p3 = [-e_p2[0], int(D*get_error(0.2))]
        e_p4 = [int(D*get_error(0.2)), int(D*get_error(0.2))]
        e_p5 = [int(D*get_error(0.2)), int(D*get_error(0.2))]
        e_p6 = [-e_p5[0], int(D*get_error(0.2))]
        e_p7 = [int(D*get_error(0.2)), int(D*get_error(0.2))]
        e_p8 = [int(D*get_error(0.2)), int(D*get_error(0.2))]
        e_p9 = [-e_p8[0], int(D*get_error(0.2))]

        p0 += e_p0
        p1 += e_p1
        p2 += e_p2
        p3 += e_p3
        p4 += e_p4
        p5 += e_p5
        p6 += e_p6
        p7 += e_p7
        p8 += e_p8
        p9 += e_p9

        points: NPNDrawPoints = {
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6,
            'p7': p7,
            'p8': p8,
            'p9': p9,
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

        params: NPNDrawParams = {
            'points': points,
            'size': (W, H)
        }

        return params

    def _get_keypoints(self, params: NPNDrawParams):
        p = params['points']

        base = p['p0']
        collector = p['p6']
        emitter = p['p9']

        kpts = {
            'base': base,
            'collector': collector,
            'emitter': emitter
        }

        return kpts

    def _draw_arrow_head(self, draw: PImageDraw, coord: npt.NDArray[np.int_]) -> None:
        D = self.step

        ah0 = coord

        u01 = get_angle_uvector(360 - (180 - 75))
        u02 = get_angle_uvector(360 - (180 - 15))

        ah1 = (ah0 + u01*D).astype(int)
        ah2 = (ah0 + u02*D).astype(int)

        e_ah0 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_ah1 = [int(D*get_error(0.1)), int(D*get_error(0.1))]
        e_ah2 = [int(D*get_error(0.1)), int(D*get_error(0.1))]

        ah0 += e_ah0
        ah1 += e_ah1
        ah2 += e_ah2

        ah0 = tuple(ah0)
        ah1 = tuple(ah1)
        ah2 = tuple(ah2)

        draw.line([ah0, ah1], fill=(255, 255, 255), width=self.draw_width)
        draw.line([ah0, ah2], fill=(255, 255, 255), width=self.draw_width)
        draw.line([ah1, ah2], fill=(255, 255, 255), width=self.draw_width)

    def _draw_image(self, params: NPNDrawParams) -> PImage:
        img = Image.new(mode='RGB', size=params['size'])

        draw = ImageDraw.Draw(img)

        p = params['points']

        draw.line([tuple(p['p0']), tuple(p['p1'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(p['p2']), tuple(p['p3'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(p['p4']), tuple(p['p5'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(p['p5']), tuple(p['p6'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(p['p7']), tuple(p['p8'])],
                  fill=(255, 255, 255), width=self.draw_width)
        draw.line([tuple(p['p8']), tuple(p['p9'])],
                  fill=(255, 255, 255), width=self.draw_width)

        self._draw_arrow_head(draw, p['p8'])

        return img


if __name__ == '__main__':
    generator = NPNGenerator()

    img, kpts = generator.generate(tgt_size=(200, 200))

    img_kpts = append_img_kpts(img, kpts)

    plt.imshow(img_kpts)
    plt.show()
