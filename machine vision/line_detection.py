import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import math_func as mf
import image_crop as ic

# image transform functions
def apply_thresholds(gray, thresholds, adaptiveSettings):
    mask1 = cv.bitwise_not(cv.threshold(gray, thresholds[0], thresholds[1], cv.THRESH_OTSU)[1])
    mask = cv.bitwise_not(cv.adaptiveThreshold(gray, thresholds[0], cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, adaptiveSettings[0], adaptiveSettings[1]))
    mask = cv.bitwise_and(mask, mask1)
    mask = cv.inRange(mask, 200, 255)
    return mask

def apply_canny(mask):
    return cv.Canny(mask, 110, 125, apertureSize=7, L2gradient=True)

# image processing
filename = "machine vision/20240802_080510.jpg"

img = cv.imread(filename)
result = ic.detect_drawing_page(img)
precence = ic.detect_drawing(result)[1]
img = ic.detect_drawing(result)[0][0]

# gray = skeletonize_image(img)
# blured = cv.GaussianBlur(gray, (3, 3), 2)
# thr = apply_thresholds(blured, (0, 255), (11, 2))
# canny = apply_canny(thr)
# cv.floodFill(canny,None,(0,0),255)
# cv.floodFill(canny,None,(0,0),0)
# dilation = cv.dilate(canny, rect_kernel, iterations = 1).astype(np.uint8)
# lines = mf.get_lines(canny)
# lines_ = [] 

# image vizualisation
"""if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lines_.append(mf.Line((x1, y1), (x2, y2), math.atan2(y2 - y1, x2 - x1)))

mf.correct_angles(lines_)
lines_, ignored = mf.reduce_lines(3, lines_, blured, 200 , 10) 

new_img = np.zeros((600,800,3), dtype=np.uint8)

for line in lines_:
    pass
    mf.draw_line(new_img, line, " Line "+str(lines_.index(line)))
"""

cv.imshow("threshold", img)
# cv.imshow("canny", thr)
# cv.imshow("thr", canny)
cv.waitKey(0)
