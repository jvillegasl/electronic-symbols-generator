from PIL import Image
from abc import ABC, abstractmethod

from models.keypoints import Keypoints


class Generator(ABC):
    @abstractmethod
    def generate(self) -> tuple[Image.Image, Keypoints]:
        pass
