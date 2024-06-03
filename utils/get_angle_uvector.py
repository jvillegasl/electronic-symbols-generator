import numpy as np
import numpy.typing as npt


def get_angle_uvector(angle: float) -> npt.NDArray[np.float_]:
    """
    Args:
        - angle (float): Angle in degrees.
    """
    rads = np.deg2rad(angle)

    x = np.cos(rads)
    y = np.sin(rads)

    uvector = np.array([x, y])

    return uvector
