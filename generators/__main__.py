import random
from PIL import Image
from PIL.Image import Image as PImage
import matplotlib.pyplot as plt

from generators.diode_generator import DiodeGenerator
from generators.ground_generator import GroundGenerator
from generators.npn_generator import NPNGenerator
from generators.vsource_generator import VSourceGenerator
from models.generator import Generator
from utils.append_img_kpts import append_img_kpts
from utils.generate_image_grid import generate_image_grid


def main():
    GENERATOR_CLASSES: list[type[Generator]] = [
        DiodeGenerator,
        GroundGenerator,
        NPNGenerator,
        VSourceGenerator,
    ]

    SAMPLES_PER_GENERATOR = 5

    samples: list[PImage] = []
    for generator_class in GENERATOR_CLASSES:
        generator = generator_class(step=500)

        for _ in range(SAMPLES_PER_GENERATOR):
            img, kpts = generator.generate(tgt_size=(500, 500))
            img_kpts = append_img_kpts(img, kpts)
            samples.append(img_kpts)

    rows = len(GENERATOR_CLASSES)
    cols = SAMPLES_PER_GENERATOR

    grid_samples = generate_image_grid(samples, rows=rows, cols=cols)

    plt.imshow(grid_samples)
    plt.show()


# def main():
#     images: list[PImage] = []

#     for _ in range(15):
#         R = random.randint(0, 255)
#         G = random.randint(0, 255)
#         B = random.randint(0, 255)

#         color = (R, G, B)
#         img = Image.new('RGB', (64, 64), color)
#         images.append(img)

#     grid = generate_image_grid(images, rows=5, cols=3)

#     plt.imshow(grid)
#     plt.show()


if __name__ == '__main__':
    main()
