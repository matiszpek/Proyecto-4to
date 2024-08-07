import numpy as np
import cv2 as cv
import imutils
import math_func as mf
import math
from typing import Union, Optional, Tuple

def detect_drawing_page(img: cv.typing.MatLike, pros_res: tuple[int, int] = (640, 480), inverted: bool = False, res: tuple[int, int] = (1080, 720)) -> Tuple[cv.typing.MatLike, cv.typing.MatLike]:
    """crops in to just the main page"""
    _img = cv.resize(img, pros_res)
    gray = cv.cvtColor(_img,cv.COLOR_BGR2GRAY)
    gray = cv.convertScaleAbs(gray, None, 1.2, -10)
    gray = cv.GaussianBlur(gray, (9, 9), 0)
    gray = cv.Canny(gray, 150, 200, apertureSize=5, L2gradient=True)
    lines = cv.HoughLinesP(gray, 1, np.pi/90, 100, minLineLength=60, maxLineGap=100)
    lines_ = [] 

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lines_.append(mf.Line((x1, y1), (x2, y2), math.atan2(y2 - y1, x2 - x1)))

    for line in lines_:
        p1, p2, mp, type, ang = line.get()
        x1, y1 = p1
        x2, y2 = p2
        cv.line(gray, (x1, y1), (x2, y2), 255, 1)

    cnts = cv.findContours(gray.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    posible_screenCnt = []
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            posible_screenCnt.append(approx)
            screenCnt = approx
    if len(posible_screenCnt) == 0:
        return img
    elif len(posible_screenCnt) > 1:
        avrg_ligth = []
        for screenCnt in posible_screenCnt:
            avrg_ligth.append(mf.get_avrg_ligth(screenCnt, img))

        screenCnt = posible_screenCnt[avrg_ligth.index(max(avrg_ligth))]

    pts1 = [(screenCnt[3][0][0], screenCnt[3][0][1]), 
            (screenCnt[2][0][0], screenCnt[2][0][1]), 
            (screenCnt[0][0][0], screenCnt[0][0][1]), 
            (screenCnt[1][0][0], screenCnt[1][0][1])]
    pts2 = [[0, 0], [pros_res[0], 0], [0, pros_res[1]], [pros_res[0], pros_res[1]]]
    if inverted:
        pts2 = mf.shift_array(pts2, 2) # rotate
    matrix = cv.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    p_result = cv.warpPerspective(_img, matrix, pros_res)

    pts1 = [mf.transform_cordinate_frame(pts1[0], pros_res, res),
            mf.transform_cordinate_frame(pts1[1], pros_res, res),
            mf.transform_cordinate_frame(pts1[2], pros_res, res),
            mf.transform_cordinate_frame(pts1[3], pros_res, res)]
    pts2 = [[0, 0], [res[0], 0], [0, res[1]], [res[0], res[1]]]
    if inverted:
        pts2 = mf.shift_array(pts2, 2) # rotate
    matrix = cv.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    b_result = cv.warpPerspective(img, matrix, res)



    return b_result, p_result

def detect_drawing(det_img: cv.typing.MatLike, cut_img: Optional[cv.typing.MatLike] = None) -> tuple[list[cv.typing.MatLike], cv.typing.MatLike]:
    """detects the drawings in the image, returns each drawing and the presence map"""

    gray = cv.cvtColor(det_img.copy(), cv.COLOR_BGR2GRAY)
    _img = cv.convertScaleAbs(gray, None, 1.6, -50)
    blured = cv.GaussianBlur(_img, (9, 9), 0) 
    _img = cv.adaptiveThreshold(blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 2)
    _img = cv.GaussianBlur(_img, (25, 25), 4) 
    _img = cv.adaptiveThreshold(blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    _img = cv.bitwise_not(_img)

    precence_map = np.zeros(_img.shape, dtype=np.float64)
    for i in range(0, _img.shape[0], 2):
        for j in range(0, _img.shape[1], 2):
            precence_map[i-10:i+10, j-10:j+10] += _img[i, j]/255

    precence_map = cv.normalize(precence_map, None, 0, 255, cv.NORM_MINMAX)
    precence_map = cv.GaussianBlur(precence_map, (25, 25), 0)
    precence_map = cv.inRange(precence_map, int(precence_map.max())*0.4, 255)
    precence_map = cv.dilate(precence_map, cv.getStructuringElement(cv.MORPH_RECT, (15, 15)), iterations=2)

    contours = cv.findContours(precence_map, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:5]
    
    new_imgs = []

    #draw contours
    for i, c in enumerate(contours):
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv.boundingRect(approx)
        if cut_img is not None:
            hi1, wi1, chn1 = det_img.shape
            hi2, wi2, chn2 = cut_img.shape
            x, y = mf.transform_cordinate_frame((x,y), (wi2, hi2), (wi1, hi1))
            w, h = mf.transform_cordinate_frame((w,h), (wi2, hi2), (wi1, hi1))
            new_imgs.append(cut_img[y:y+h, x:x+w])
        else:
            new_imgs.append(det_img[y:y+h, x:x+w])

    return new_imgs, precence_map

# test section
if __name__ == "__main__":
    filename = "machine vision/20240802_080510.jpg"
    img = cv.imread(filename)
    result = detect_drawing_page(img, res= (1754, 1240))
    precence = detect_drawing(result[1])[1]
    results = detect_drawing(result[1], result[0])[0]

    cv.imshow('result', result[0])
    for i in range(len(results)):
        cv.imshow(f'result {i}', results[i])
    cv.imshow('presence', precence)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
