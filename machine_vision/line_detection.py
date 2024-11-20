import cv2 as cv
import numpy as np
import math
import math_func as mf
import image_crop as ic
from tqdm import tqdm
import imutils

# image processing
filename = "machine_vision/prueba_cubo2.jpg"
img_ = cv.imread(filename)
img_ = imutils.rotate_bound(img_, -90)
result = ic.detect_drawing_page(img_, (1080, 720), False, (842, 595))
imges = ic.detect_drawing(result)[0]
def get_drawing(img, name: str = ""): 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    or_mask = mf.get_img_contrast(gray, 0.995, 7, 2)
    cv.imshow("mask"+str(name), or_mask)
    mask = cv.morphologyEx(or_mask, cv.MORPH_CLOSE, np.ones((3,3),np.uint8))
    # mask = cv.GaussianBlur(mask, (9,9), 3)
    # mask = cv.bilateralFilter(mask, 7, 90, 90)

    def get_lines(img: np.ndarray = mask) -> list:
        lines = cv.HoughLinesP(img, 2, np.pi/360, 80, minLineLength=30, maxLineGap=3)
        lines = np.concatenate([cv.HoughLinesP(img, 2, np.pi/360, 50, minLineLength=3, maxLineGap=1), lines])

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
                
                if y1 < y2:
                    y = y1
                    x = x1
                    y1 = y2
                    x1 = x2
                    y2 = y
                    x2 = x

                if y1 == y2:
                    angle = 0
                else:
                    angle = math.atan2((y2 - y1), (x2 - x1))

                    if abs(angle) > np.pi/2:
                        angle = angle - np.pi
                    elif abs(angle) <= -np.pi/2:
                        angle = angle + np.pi

                lines_.append(mf.Line((x1, y1), (x2, y2), normal = angle))
        return lines_

    new_img = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8) 
    lines_ = get_lines(mask)
    for i, line in enumerate(lines_):
        mf.draw_line(new_img, line, "", (255/int(len(lines_))*i, 255, 100))
        # cv.putText(new_img, str(i), (line.start[0], line.start[1]), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("line"+str(name), new_img)


    # image vizualisation

    
    img_complexity = mf.get_img_complexity(gray, 
                                        0.5, 
                                        0.9, 
                                        9, 
                                        .5, 
                                        np.pi/720, 
                                        2, 
                                        10, 
                                        10)
    new_img = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8) 
    img_complexity = cv.resize(img_complexity, img.shape[:2][::-1])

    i = 0
    for n in range(1):
        lines_, edges, noise = mf.group_lines(lines_, 3, 7, math.pi/10)
        lines = []
        for group in lines_:
            line = mf.join_line_group(group[1:])
            lines.append(line)
        lines_ = lines   
        

    for line in lines_:
        for line_ in lines_:
            if line != line_:
                if mf.check_lines_same(line, line_, 5, 0.5, 2):
                    lines_.remove(line_)

    print(len(lines_))

    for i, line in enumerate(lines_):
        mf.draw_line(new_img, line, "", (255/int(len(lines_))*i, 255, 100))
        line.debug()
        # cv.putText(new_img, str(i), (line.start[0], line.start[1]), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv.LINE_AA)

    points = mf.get_vertices(lines_) 
    # print(points)
    for p in points:
        cv.circle(new_img, p[0], 5, (255, 255, 255))     
    cv.imshow(name, new_img)

# new_img = cv.addWeighted(new_img, 0.7, mask, 0.3, 0)

# new_img = cv.morphologyEx(new_img, cv.MORPH_CLOSE, np.ones((2,2),np.uint8))

for i in range(3):
    get_drawing(imges[i], str(i))
cv.waitKey(0)