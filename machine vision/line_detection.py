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
dilation = cv.dilate(gray, rect_kernel, iterations = 1).astype(np.uint8)
contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

thr = apply_thresholds(dilation, (0, 255), (11, 2))
thr = cv.GaussianBlur(thr, (3, 3), 2)
canny = apply_canny(thr)
dilation = cv.dilate(thr, rect_kernel, iterations = 1).astype(np.uint8)
lines = get_lines(dilation)
lines_ = []

# image vizualisation
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    
    # Drawing a rectangle on copied image
    rect = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lines_.append(mf.Line((x1, y1), (x2, y2), math.atan2(y2 - y1, x2 - x1)))

lines_, ignored = mf.reduce_lines(1, lines_, img, 200, 3) # still needs work
  
for line in lines_:
    pass
    mf.draw_line(img, line, " Line "+str(lines_.index(line)))


cv.imshow("threshold", img)
# cv.imshow("canny", canny)
# cv.imshow("thr", thr)
cv.waitKey(0)
