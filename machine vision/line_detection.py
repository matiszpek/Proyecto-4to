import cv2 as cv
import numpy as np
import math
import math_func as mf
import image_crop as ic
from tqdm import tqdm

# image processing
filename = "machine vision/20240802_080510.jpg"

img_ = cv.imread(filename)
result = ic.detect_drawing_page(img_, res= (1080, 720))
img = ic.detect_drawing(result)[0][0]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def get_img_contrast(img: cv.typing.MatLike, ) -> cv.typing.MatLike:  
    inv = 255 - img
    blured = cv.GaussianBlur(inv, (9, 9), 1)
    inv_blured = 255 - blured
    img = cv.divide(img, inv_blured, scale=256)
    mask = cv.inRange(img, 0, int(np.max(img)*0.975))
    return mask

# image complexity
def get_img_complexity(img: cv.typing.MatLike, ) -> cv.typing.MatLike:
    
    mask = get_img_contrast(img)
    lines = cv.HoughLinesP(mask, .25, np.pi/360, 2, minLineLength=10, maxLineGap=10)
    lines_ = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lines_.append(mf.Line((x1, y1), (x2, y2), math.atan2(y2 - y1, x2 - x1)))
    img_complexity = mf.get_line_presence(lines_, img, (img.shape[0]/2, img.shape[1]/2))
    img_complexity = cv.GaussianBlur(img_complexity, (9, 9), 0)
    img_complexity = cv.inRange(img_complexity, int(np.max(img_complexity)*0.9), 255)
    return img_complexity

mask = get_img_contrast(gray)
lines = cv.HoughLinesP(mask, 1, np.pi/180, 20, minLineLength=20, maxLineGap=3)

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

new_img = np.zeros((600,800,3), dtype=np.uint8)

# image vizualisation
img_complexity = get_img_complexity(gray)
img_complexity = cv.resize(img_complexity, img.shape[:2][::-1])
# mask = cv.addWeighted(cv.bitwise_not(img_complexity), 0.5, mask, 0.5, 0)

for line in tqdm(lines_):
    pix = mf.scan_line_pixels(line, cv.GaussianBlur(mask, (3, 3), 0))
    certanty = np.mean(np.array(pix)) 
    if certanty > 120:
        mf.draw_line(new_img, line, "", (255, certanty, 255))

        
img_complexity = cv.cvtColor(img_complexity, cv.COLOR_GRAY2BGR)
img_complexity[:,:,0] = 0
img_complexity[:,:,2] = 0
img = cv.addWeighted(img, 0.5, img_complexity, 0.5, 0)

cv.imshow("threshold", mask)
cv.imshow("canny", img_complexity)
cv.imshow("thr", new_img)
cv.imshow("original", img)
cv.waitKey(0)