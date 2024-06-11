import math
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PImage

from models.keypoints import Keypoints
from utils.get_random_color import get_random_color


def append_img_kpts(img: PImage, kpts: Keypoints) -> PImage:
    new_img = img.copy()

    W, H = new_img.size

    RADIUS = 0.01 * min([W, H])
    colors = {
        k: tuple([int(255 * t) for t in get_random_color()])
        for k in kpts.keys()
    }

    draw = ImageDraw.Draw(new_img, 'RGBA')

    for cls in kpts:
        coords = kpts[cls]
        color = colors[cls]

        d = RADIUS * math.sqrt(2)
        top_left = (coords - d).astype(int)
        bottom_right = (coords + d).astype(int)

        top_left = tuple(top_left)
        bottom_right = tuple(bottom_right)

        draw.ellipse([top_left, bottom_right], fill=color)

        bbox_color = color + (128,)
        font = ImageFont.truetype("arial.ttf", int(10*RADIUS))

        text_bbox = draw.textbbox(
            xy=top_left, text=cls, align='left', font=font)

        d_text_x = text_bbox[2] - W
        d_text_y = text_bbox[3] - H

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        fit_x = text_bbox[0] - max(d_text_x, 0)
        fit_y = text_bbox[1] - max(d_text_y, 0)

        fit_text_bbox = (fit_x, fit_y, fit_x + text_width, fit_y + text_height)

        draw.rectangle(fit_text_bbox, fill=bbox_color)
        draw.text(xy=(fit_x, fit_y), text=cls, fill='black', font=font)

    return new_img
