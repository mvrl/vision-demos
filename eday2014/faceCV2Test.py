import cv2
import cv2.cv as cv
import numpy as np

if __name__ == "__main__":

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    c = cv2.VideoCapture(1)

    while(1):
        _,im = c.read()

        faces = face_cascade.detectMultiScale(im, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),1)
       
        cv2.flip(im, 1, im)

        cv2.imshow('e2', im)

        k = cv2.waitKey(1)
        
        if k == 0x1b:
            break
        
    c.release()    
    cv2.destroyAllWindows()
