from io import BytesIO
from urllib.request import urlopen
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

# url = "https://i.ytimg.com/vi/W4qijIdAPZA/maxresdefault.jpg"
# file = BytesIO(urlopen(url).read())
# img = Image.open(file)
# draw = ImageDraw.Draw(img, "RGBA")
# draw.rectangle(((280, 10), (1010, 706)), fill=(200, 100, 0, 0))
# # draw.rectangle(((280, 10), (1010, 706)), outline=(0, 0, 0, 127), width=3)
# # img.save('orange-cat.png')
# plt.imshow(img)
# plt.show()

img = Image.new('RGB', (1280, 800), (255, 0, 0))
draw = ImageDraw.Draw(img, "RGBA")
draw.rectangle(((280, 10), (1010, 706)), fill=(200, 100, 0, 128))
plt.imshow(img)
plt.show()
