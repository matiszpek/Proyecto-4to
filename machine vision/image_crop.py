import numpy as np
import cv2 as cv
import imutils
import math_func as mf
import math

def detect_drawing_page(img: cv.typing.MatLike, pros_res: tuple[int, int] = (640, 480)) -> cv.typing.MatLike:
    """crops in to just the main page"""
    img = cv.resize(img, pros_res)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
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

    pts1 = [(screenCnt[3][0][0], screenCnt[3][0][1]), (screenCnt[2][0][0], screenCnt[2][0][1]), (screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[1][0][0], screenCnt[1][0][1])]
    pts2 = [[0, 0], [640, 0], [0, 480], [640, 480]]
    pts2 = mf.shift_array(pts2, 0) # rotate
    
    matrix = cv.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    result = cv.warpPerspective(img, matrix, (640, 480))

    return result

def detect_drawing(img: cv.typing.MatLike) -> list[cv.typing.MatLike]:
    detector = cv.SimpleBlobDetector()
    keypoints = detector.detect(img)
    _img = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return [_img]


if __name__ == "__main__":
    filename = "machine vision/20240802_080556.jpg"
    img = cv.imread(filename)
    result = detect_drawing_page(img)
    result = detect_drawing(result)[0]
    cv.imshow('result', result)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

