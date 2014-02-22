import cv2
import cv2.cv as cv
import numpy as np

global src_pts
global count
count = 0
src_pts = cv2.cv.CreateMat(4,1,cv2.CV_64FC2)

def getPoint(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global src_pts
        global count
        src_pts[count,0] = (x,y)
        count = count + 1

baseIm = cv2.imread('P1010580.jpg')

cv2.namedWindow('Calibrate')

cv.SetMouseCallback('Calibrate',getPoint)

cv2.imshow('Calibrate',baseIm)

cv2.waitKey(0)

if count==4:
    cv2.destroyAllWindows()

dest_pts = cv2.cv.CreateMat(4,1,cv2.CV_64FC2)

#src_pts[0,0] = (392,144)
#src_pts[1,0] = (592,237)
#src_pts[2,0] = (272,322)
#src_pts[3,0] = (494,436)

dest_pts[0,0] = (0,0)
dest_pts[1,0] = (520,0)
dest_pts[2,0] = (0,520)
dest_pts[3,0] = (520,520)

src_pts = np.array(src_pts)
dest_pts = np.array(dest_pts)

M,mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, 5.0)

#dst = cv2.perspectiveTransform(baseIm,M)

output = cv2.warpPerspective(baseIm,M,(900,900))

cv2.imshow('base',baseIm)
cv2.imshow('output',output)

k = cv2.waitKey(0)

