from PIL import Image

for i in range(1, 4):
    img = Image.new('RGB', (640, 480), color=(73, 109, 137))
    img.save(f'img{i}.jpg')