import cv2
import cv2.cv as cv
import numpy as np

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
    #gray = cv2.medianBlur(gray,7)
    
    return im,gray

# Detects faces in the passed in det image and
# draws the box around the detected face on the
# passed in im image
# Det - blurred grayscale image
# Im - normal frame from the webcam
def detectFaces(det,im):
    # Detect faces in the det image
    faces = face_cascade.detectMultiScale(det, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)

    # Draw a blue box around the detected faces in the im image
    for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),1)

# Detects circles in the passed in det image
# and draws the center of the circle and the
# circle itself on the passed in im image
# Det - blurred grayscale image
# Im - normal frame from the webcam
def detectCircles(det,im):
    # Detect circles in the det image
    circles = cv2.HoughCircles(det, cv.CV_HOUGH_GRADIENT, 1.5, 20, param1=200, param2=100, minRadius=0, maxRadius=0)
    #circles = np.uint16(np.around(circles))

    # Draws circles on the im image. If there are no circles, an
    # error is thrown, but the try-catch handles the errors.
    try:
        for i in circles[0,:10]:
            cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(im,(i[0],i[1]),2,(0,0,255),3)
    except (ValueError, TypeError):
        pass

if __name__ == "__main__":
    # Path to the Haar Cascade file
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

    # Main loop to detect faces/circles on the webcam stream
    while(1):
        # Set up images
        im,gray = setupImages(c)

        #cv2.flip(im, 1, im)

        im = cv2.warpPerspective(im,M,(900,900))
        gray = cv2.warpPerspective(gray,M,(900,900))

        im = im[0:460, 0:940]
        gray = gray[0:460, 0:940]

        # Detect Faces
        #detectFaces(gray,im)

        # Detect Circles
        detectCircles(gray,im)

        # Flip the image horizontally and display it
        
        cv2.imshow('e2', im)

        # Wait until the escape key is pressed
        # Break the loop when it is pressed
        k = cv2.waitKey(1)
        if k == 0x1b:
            break

    # Close all windows
    c.release()    
    cv2.destroyAllWindows()
