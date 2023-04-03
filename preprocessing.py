import cv2
from mtcnn import MTCNN
from PIL import Image

import os

# folder path
count = 0

dir_path = f"resources/image-source/testing-source-for-github"

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)

detector = MTCNN()
# for i in range(1, count+1):
for i in range(17,18):
    img_path = f"{dir_path}/original/me-{i}.JPG"
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    bounding_box = result[0]['box']

    x = bounding_box[0]
    y = bounding_box[1]
    width = bounding_box[2]
    height = bounding_box[3]
    mid_point_x = x + width/2
    mid_point_y = y + height/2

    if width > height:
        side = width
        y = mid_point_y - side/2
    else:
        side = height
        x = mid_point_x - side/2

    left = x
    top = y
    right = x + side
    bottom = y + side

    im = Image.open(img_path)

    im_crop = im.crop((left, top, right, bottom))
    imResize = im_crop.resize((250, 250), Image.ANTIALIAS)
    imResize.save(f"{dir_path}/cropped/cropped-me-{i}.jpg", "jpeg")
    # imResize.show()
