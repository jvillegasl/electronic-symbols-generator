from typing import Any, Optional
from PIL.Image import Image as PImage
from PIL.Image import Resampling
from PIL import Image
from abc import ABC, abstractmethod

from models.keypoints import Keypoints


class Generator(ABC):
    step: int

    def __init__(self, step: int = 100):
        if step is not None:
            self.step = step

    @property
    def draw_width(self) -> int:
        return max(2, int(self.step/25))

    @abstractmethod
    def _get_draw_params(self) -> Any:
        pass

    @abstractmethod
    def _draw_image(self, params) -> PImage:
        pass

    @abstractmethod
    def _get_keypoints(self, params) -> Keypoints:
        pass

    def _fit_scale(self, tgt_size: tuple[int, int], img: PImage, kpts: Keypoints) -> tuple[PImage, Keypoints]:
        W, H = img.size
        RATIO = W/H

        tgt_W, tgt_H = tgt_size

        if tgt_H*RATIO <= tgt_W:
            scaled_W = int(tgt_H*RATIO)
            scaled_H = tgt_H
        else:
            scaled_W = tgt_W
            scaled_H = int(scaled_W/RATIO)

        scale_ratio = scaled_W / W
        scaled_size = (scaled_W, scaled_H)
        scaled_img = img.resize(scaled_size)

        scaled_W, scaled_H = scaled_size
        paste_coords = (int((tgt_W - scaled_W) / 2),
                        int((tgt_H - scaled_H) / 2))

        new_img = Image.new('RGB', tgt_size, (0, 0, 0))
        new_img.paste(scaled_img, paste_coords)

        scaled_kpts: Keypoints = {}
        for key in kpts:
            keypoint = kpts[key]
            scaled_keypoint = scale_ratio * keypoint
            scaled_keypoint += paste_coords

            scaled_kpts[key] = scaled_keypoint.astype(int)

        return new_img, scaled_kpts

    def generate(self, tgt_size: Optional[tuple[int, int]] = None) -> tuple[PImage, Keypoints]:
        params = self._get_draw_params()

        img = self._draw_image(params)

        kpts = self._get_keypoints(params)

        if tgt_size is not None:
            img, kpts = self._fit_scale(tgt_size, img, kpts)

        return img, kpts
