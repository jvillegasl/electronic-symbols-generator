import random


def get_error(error_pct: float, absolute=False):
    e = error_pct * random.uniform(-1, 1)

    if absolute:
        return abs(e)
    else:
        return e
