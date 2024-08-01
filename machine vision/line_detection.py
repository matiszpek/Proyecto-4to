import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import math_func as mf

# image transform functions
def apply_thresholds(gray, thresholds, adaptiveSettings):
    mask1 = cv.bitwise_not(cv.threshold(gray, thresholds[0], thresholds[1], cv.THRESH_OTSU)[1])
    mask = cv.bitwise_not(cv.adaptiveThreshold(gray, thresholds[0], cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, adaptiveSettings[0], adaptiveSettings[1]))
    mask = cv.bitwise_and(mask, mask1)
    mask = cv.inRange(mask, 200, 255)
    return mask

def apply_canny(mask):
    return cv.Canny(mask, 110, 125, apertureSize=7, L2gradient=True)

def get_lines(can):
    return cv.HoughLinesP(
        can,
        0.5,
        np.pi/8,
        10,
        minLineLength=5,
        maxLineGap=10
    )

# image processing
img = cv.imread("machine vision/technical_drawing_sample0.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))

blured = cv.GaussianBlur(gray, (3, 3), 2)
thr = apply_thresholds(blured, (0, 255), (11, 2))
canny = apply_canny(thr)
cv.floodFill(canny,None,(0,0),255)
cv.floodFill(canny,None,(0,0),0)
# dilation = cv.dilate(canny, rect_kernel, iterations = 1).astype(np.uint8)
lines = get_lines(canny)
lines_ = [] 

# image vizualisation
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lines_.append(mf.Line((x1, y1), (x2, y2), math.atan2(y2 - y1, x2 - x1)))

mf.correct_angles(lines_)
lines_, ignored = mf.reduce_lines(3, lines_, blured, 200 , 10) 

new_img = np.zeros((600,800,3), dtype=np.uint8)

for line in lines_:
    pass
    mf.draw_line(new_img, line, " Line "+str(lines_.index(line)))


cv.imshow("threshold", new_img)
cv.imshow("canny", thr)
cv.imshow("thr", canny)
cv.waitKey(0)
