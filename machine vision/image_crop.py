import numpy as np
import cv2 as cv
import imutils
# import math_func as mf

# resources
# https://stackoverflow.com/questions/6555629/algorithm-to-detect-screenCnt-of-paper-sheet-in-photo
# https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html

filename = "machine vision/20240802_080527.jpg"
img = cv.imread(filename)
img = cv.resize(img, (640 , 480))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (9, 9), 0)
gray = cv.Canny(gray, 110, 200, apertureSize=5, L2gradient=True)
cnts = cv.findContours(gray.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]

for c in cnts:
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)
	
	if len(approx) == 4:
		screenCnt = approx
		break

pts1 = [(screenCnt[3][0][0], screenCnt[3][0][1]), (screenCnt[2][0][0], screenCnt[2][0][1]), (screenCnt[0][0][0], screenCnt[0][0][1]), (screenCnt[1][0][0], screenCnt[1][0][1])]
pts2 = [[0, 0], [0, 480], [640, 0], [640, 480]]
matrix = cv.getPerspectiveTransform(np.float32(screenCnt), np.float32(pts2))
result = cv.warpPerspective(img, matrix, (640, 480))


cv.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
# img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst', result)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()