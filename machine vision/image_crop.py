import numpy as np
import cv2 as cv
import imutils
# import math_func as mf

# resources
# https://stackoverflow.com/questions/6555629/algorithm-to-detect-corners-of-paper-sheet-in-photo
# https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html

filename = "machine vision/20240802_080527.jpg"
img = cv.imread(filename)
img = cv.resize(img, (800 , 600))
img = cv.convertScaleAbs(img, alpha = 1, beta = 20)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (9, 9), 0)
gray = cv.Canny(gray, 110, 200, apertureSize=5, L2gradient=True)
# largest ones, and initialize the screen contour
cnts = cv.findContours(gray.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
# Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst', img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()