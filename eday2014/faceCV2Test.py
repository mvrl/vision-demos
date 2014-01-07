import cv2
import cv2.cv as cv
import numpy as np

if __name__ == "__main__":

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    c = cv2.VideoCapture(1)

    while(1):
        _,im = c.read()
        #gray = im
        #cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #gray = cv2.medianBlur(gray,5)
        
        faces = face_cascade.detectMultiScale(im, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),1)

        circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1.5, 20, param1=200, param2=100, minRadius=0, maxRadius=0)
        #circles = np.uint16(np.around(circles))

        try:
            for i in circles[0,:10]:
                cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(im,(i[0],i[1]),2,(0,0,255),3)
        except (ValueError, TypeError):
            print 'no circles'
            
        cv2.flip(im, 1, im)

        cv2.imshow('e2', im)

        k = cv2.waitKey(1)
        
        if k == 0x1b:
            break
        
    c.release()    
    cv2.destroyAllWindows()
