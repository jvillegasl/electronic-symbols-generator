import random


def get_error(error_pct: float, a=False):
    e = error_pct * random.uniform(-1, 1)

    if a:
        return abs(e)
    else:
        return e
