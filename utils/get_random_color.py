import numpy as np


def get_random_color() -> tuple[float, float, float]:
    color = np.random.choice(range(256), size=3).astype(float)
    color /= 255
    color = tuple(color)

    return color
