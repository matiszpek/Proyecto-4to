import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

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
        1,
        np.pi/90,
        20,
        minLineLength=8,
        maxLineGap=5
    )

def draw_line(img_lines, line):
    x1, y1, x2, y2, angle = line
    cv.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv.putText(img_lines, " Line "+str(lines_.index(line)), (int((x2+x1)/2), int((y2+y1)/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
    cv.circle(img_lines, (x1, y1), 5, (255, 0, 0), -1)
    cv.circle(img_lines, (x2, y2), 5, (255, 255, 0), -1)

def calculate_slope_and_intercept(x1, y1, x2, y2):
    x = np.sort([x1, x2])
    y = np.sort([y1, y2])
    x1, x2 = x
    y1, y2 = y
    x2 = x2 if x1 != x2 else x2+0.5
    slope = (y2 - y1) / (x2 - x1)
    slope = slope if slope != 0 else 0.05
    b = y1 - slope * x1
    return slope, b

img = cv.imread("machine vision/technical_drawing_sample0.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
thr = apply_canny(gray)
lines = get_lines(thr)
lines_ = []



rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))
print()
# Applying dilation on the threshold image
dilation = cv.dilate(gray, rect_kernel, iterations = 20).astype(np.uint8)

# Finding contours
contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    
    # Drawing a rectangle on copied image
    rect = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lines_.append([x1, y1, x2, y2, -90+(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)])

for line in lines_:
    draw_line(img, line)

cv.imshow("threshold", img)
cv.waitKey(0)
