import cv2 as cv
import numpy as np
import math
import math_func as mf
import image_crop as ic

# image processing
filename = "machine vision/20240802_080510.jpg"

img_ = cv.imread(filename)
result = ic.detect_drawing_page(img_, res= (1080, 720))
img = ic.detect_drawing(result)[0][0]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
inv = 255 - gray
blured = cv.GaussianBlur(inv, (9, 9), 0)
inv_blured = 255 - blured
img = cv.divide(gray, inv_blured, scale=256)

# thresholding
mask = cv.inRange(img, 0, int(np.max(img)*0.975))

# hough line detection
lines = cv.HoughLinesP(mask, .5, np.pi/180, 4, minLineLength=10, maxLineGap=10)
lines_ = []

# image vizualisation
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lines_.append(mf.Line((x1, y1), (x2, y2), math.atan2(y2 - y1, x2 - x1)))

new_img = np.zeros((600,800,3), dtype=np.uint8)

for line in lines_:
    pass
    mf.draw_line(new_img, line, "")

# image complexity
img_complexity = mf.get_line_presence(lines_, img, (img.shape[0]/2, img.shape[1]/2))
img_complexity = cv.GaussianBlur(img_complexity, (9, 9), 0)
img_complexity = cv.inRange(img_complexity, int(np.max(img_complexity)*0.9), 255)

# image vizualisation
img_complexity = cv.resize(img_complexity, img.shape[:2][::-1])
img = cv.addWeighted(img, 0.5, img_complexity, 0.5, 0)
cv.imshow("threshold", img)
cv.imshow("canny", img_complexity)
cv.imshow("thr", new_img)
cv.imshow("original", gray)
cv.waitKey(0)