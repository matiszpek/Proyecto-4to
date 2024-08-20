import easyocr
import cv2 as cv
import image_crop as ic
import numpy as np

def get_img_contrast(img: cv.typing.MatLike, ) -> cv.typing.MatLike:  
    inv = 255 - img
    blured = cv.GaussianBlur(inv, (9, 9), 1)
    inv_blured = 255 - blured
    img = cv.divide(img, inv_blured, scale=256)
    mask = cv.inRange(img, 0, int(np.max(img)*0.975))
    return mask

filename = "machine vision/20240802_080539.jpg"

img_ = cv.imread(filename)
page = ic.detect_drawing_page(img_, res= (1080*2, 720*2))
img = ic.detect_drawing(page)[0][0]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = get_img_contrast(gray)
gray = cv.GaussianBlur(gray, (3, 3), 0)
gray = cv.rotate(gray, cv.ROTATE_90_CLOCKWISE)

reader = easyocr.Reader(['es'], gpu = True)
result = reader.readtext(gray)
gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
for _ in result:
    cv.drawContours(gray, [np.array(_[0], dtype = np.int32)], -1, (0, 255, 0), 2)
    print(_[1])

cv.imshow("img", gray)
cv.waitKey(0)