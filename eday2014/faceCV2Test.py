import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

c = cv2.VideoCapture(1)

while(1):
    _,im = c.read()
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),1)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = im[y:y+h, x:x+w]

    
    cv2.flip(im, 1, im)

    cv2.imshow('e2', im)
    
    if cv2.waitKey(5)==27:
        break
    
cv2.destroyAllWindows()
