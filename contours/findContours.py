import cv2
import cv2.cv as cv
import numpy as np
import cPickle

# Global variables
global src_pts
global count
count = 0
src_pts = cv2.cv.CreateMat(4,1,cv2.CV_64FC2)

# Mouse callback function
# Checks if the event is a left click
# Records the image coordinates if so
def getPoint(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global src_pts
        global count
        src_pts[count,0] = (x,y)
        count = count + 1

# Sets up the images to be processed and displayed
# Grabs the next from the webcam and converts it
# to gray scale for processing
def setupImages(c):
    # Grab next frame
    _,im = c.read()

    # Convert the image to gray scale and blur it
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,1)
    
    _,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    return im,thresh

# Detects contours in the passed in det image
# and draws the contours of the image on the
# passed in im image
# Det - blurred grayscale image
# Im - normal frame from the webcam
def detectContours(det,im):
    threshold = 132
    edges = cv2.Canny(det, threshold, threshold*2)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tmp = cPickle.dumps(contours)    
    contours = cPickle.loads(tmp)
    
    cv2.drawContours(im, contours, -1, (0,255,0), 1)

if __name__ == "__main__":
    # Opens the webcam
    c = cv2.VideoCapture(0)

    # Loop to set up the calibration
    while(1):
        # Grab next frame
        calibIm,_ = setupImages(c)

        # Set up the window and display the image
        cv2.namedWindow('Calibrate')
        cv.SetMouseCallback('Calibrate',getPoint)
        cv2.imshow('Calibrate',calibIm)

        # Wait until the user hits a key
        k = cv2.waitKey(0)
        if k == 0x1b:
            cv2.destroyAllWindows()
            break

    # Set up the destination coordinates
    dest_pts = cv2.cv.CreateMat(4,1,cv2.CV_64FC2)
    dest_pts[0,0] = (0,0)
    dest_pts[1,0] = (940,0)
    dest_pts[2,0] = (0,460)
    dest_pts[3,0] = (940,460)

    # Convert the point arrays to numpy arrays
    src_pts = np.array(src_pts)
    dest_pts = np.array(dest_pts)

    # Calculate the homography between the image and the camera
    M,mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, 5.0)

    ix = 0
    # Main loop to detect faces/circles on the webcam stream
    while(1):
        ix = ix + 1
        
        # Set up images
        im,thresh = setupImages(c)

        im = cv2.warpPerspective(im,M,(900,900))
        thresh = cv2.warpPerspective(thresh,M,(900,900))

        im = im[0:460, 0:940]
        thresh = thresh[0:460, 0:940]
        
        # Detect Circles
        detectContours(thresh,im)

        # Flip the image horizontally and display it
        
        cv2.imshow('contours', im)
        cv2.imshow('threshold', thresh)

        # Wait until the escape key is pressed
        # Break the loop when it is pressed
        k = cv2.waitKey(1)
        if k == 0x1b:
            break

    # Close all windows
    c.release()    
    cv2.destroyAllWindows()
