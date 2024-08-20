import cv2 as cv
import numpy as np
import math
import libs.math_func as mf
import image_crop as ic
from tqdm import tqdm

# image processing
filename = "machine vision/20240802_080510.jpg"

img_ = cv.imread(filename)
result = ic.detect_drawing_page(img_, res= (1080*2, 720*2))
img = ic.detect_drawing(result)[0][0]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

or_mask = mf.get_img_contrast(gray)
mask = cv.morphologyEx(or_mask, cv.MORPH_CLOSE, np.ones((3,3),np.uint8))


def get_lines():
    lines = cv.HoughLinesP(mask, 2, np.pi/180, 80, minLineLength=40, maxLineGap=3)
    lines = np.concatenate([cv.HoughLinesP(mask, 2, np.pi/180, 80, minLineLength=3, maxLineGap=1), lines])

    lines_ = []

    # image vizualisation
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < x2:
                x = x1
                y = y1
                x1 = x2
                y1 = y2
                x2 = x
                y2 = y
            
            if x1 == x2:
                angle = np.pi
            elif y1 == y2:
                angle = 0
            else:
                angle = math.atan((y2 - y1)/(x2 - x1))

                if abs(angle) > np.pi/2:
                    angle = angle - np.pi
                elif abs(angle) < np.pi/2:
                    angle = angle + np.pi

            lines_.append(mf.Line((x1, y1), (x2, y2), normal=angle))
    return lines_

lines_ = get_lines()
new_img = np.zeros((600,800,3), dtype=np.uint8) 

# image vizualisation
img_complexity = mf.get_img_complexity(gray)
img_complexity = cv.resize(img_complexity, img.shape[:2][::-1])
# mask = cv.addWeighted(cv.bitwise_not(img_complexity), 0.5, mask, 0.5, 0)

lines_ = mf.tidy_lines(lines_, mask, 5, 1.2, math.pi/30)
deleted = 0
for line in tqdm(lines_):
    mf.draw_line(new_img, line, "", (int(abs(line.normal)), 255, 255))
    

print(deleted)
        
img_complexity = cv.cvtColor(img_complexity, cv.COLOR_GRAY2BGR)
img_complexity[:,:,0] = 0
img_complexity[:,:,2] = 0
img = cv.addWeighted(img, 0.5, img_complexity, 0.5, 0)

new_img = cv.morphologyEx(new_img, cv.MORPH_CLOSE, np.ones((2,2),np.uint8))

cv.imshow("threshold", mask)
cv.imshow("canny", img_complexity)
cv.imshow("thr", new_img)
cv.imshow("original", img)
cv.waitKey(0)