from PIL import Image
from PIL.Image import Image as PImage


def generate_image_grid(imgs: list[PImage], rows: int, cols: int) -> PImage:
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid
