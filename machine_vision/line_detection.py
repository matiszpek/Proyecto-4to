import cv2 as cv
import numpy as np
import math
import math_func as mf
import image_crop as ic
from tqdm import tqdm

# image processing
filename = "machine_vision/20240802_080510.jpg"

img_ = cv.imread(filename)
result = ic.detect_drawing_page(img_, res= (1080*2, 720*2))
img = ic.detect_drawing(result)[0][0]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

or_mask = mf.get_img_contrast(gray)
mask = cv.morphologyEx(or_mask, cv.MORPH_CLOSE, np.ones((3,3),np.uint8))
# mask = cv.GaussianBlur(mask, (9,9), 3)
# mask = cv.bilateralFilter(mask, 7, 90, 90)

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
# lines_ = mf.tidy_lines(lines_, cv.dilate(mask, (5,5), iterations=1), 9, 1.2, math.pi/30)


# image vizualisation

new_img = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8) 
img_complexity = mf.get_img_complexity(gray)
img_complexity = cv.resize(img_complexity, img.shape[:2][::-1])

for line in tqdm(lines_):
    mf.draw_line(new_img, line, "", (int(abs(line.normal)), 255, 255))
        
img_complexity = cv.cvtColor(img_complexity, cv.COLOR_GRAY2BGR)
img_complexity[:,:,0] = 0  
img_complexity[:,:,1] = 0
#mask = cv.dilate(mask, (5,5), iterations=1)
mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
mask[:,:,1] = 0
mask[:,:,0] = 0
new_img = cv.addWeighted(new_img, 0.7, mask, 0.3, 0)

# new_img = cv.morphologyEx(new_img, cv.MORPH_CLOSE, np.ones((2,2),np.uint8))

cv.imshow("threshold", mask)
cv.imshow("canny", img_complexity)
cv.imshow("thr", new_img)
cv.imshow("original", img)
cv.waitKey(0)