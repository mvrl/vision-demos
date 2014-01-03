import cv2
import numpy as np
import math
import time

#
# Shows two video streams at the same time
#
# if you get memory errors (especially on linux), make sure your cameras
# are on different USB hubs
#

if __name__ == "__main__":

  # open the camera
  c1 = cv2.VideoCapture(0)
  time.sleep(2)
  c2 = cv2.VideoCapture(1)

  _,im1 = c1.read()
  time.sleep(2)
  _,im2 = c2.read()

  while True:

    _,im1 = c1.read()
    _,im2 = c2.read()

    cv2.imshow("Camera 1",im1) 
    cv2.imshow("Camera 2",im2)
    
    # handle user input
    k = cv2.waitKey(1)

    if k == 0x1b:
      print 'ESC pressed.  Exiting infinite loop.'
      break

  c1.release()
  c2.release()
  cv2.destroyAllWindows()

