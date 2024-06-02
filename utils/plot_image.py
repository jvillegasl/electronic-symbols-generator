import matplotlib.pylab as plt
from PIL.Image import Image as PImage

from models.keypoints import Keypoints
from utils.get_random_color import get_random_color


def plot_image(img: PImage, kpts: Keypoints):
    W, H = img.size

    ax: plt.Axes
    _, ax = plt.subplots()

    plt.imshow(img)

    radius = 0.01 * min([W, H])
    colors = {k: get_random_color() for k, v in kpts.items()}

    for cls in kpts:
        coords = tuple(kpts[cls])
        color = colors[cls]

        circle = plt.Circle(coords, radius, color=color)
        ax.add_artist(circle)

        t = ax.text(coords[0] + 4 * radius, coords[1] - 4 * radius, cls)
        t.set_bbox({'facecolor': color, 'alpha': 1})

    plt.show()
