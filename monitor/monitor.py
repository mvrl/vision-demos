import cv2
import numpy as np
import math

#
# a very simple example of python background subtraction
#
# Limitations: 
#  - I can't progromatically control the autoexposure
#

if __name__ == "__main__":

  # open the camera
  camera = cv2.VideoCapture(1)

  bgModel = cv2.BackgroundSubtractorMOG(history=500,nmixtures=6,backgroundRatio=.1,noiseSigma=5) 

  while True:

    # grab an image from the camera
    _,image = camera.read()

    fg = bgModel.apply(image,None, .01)

    cv2.imshow("Live Video",image) 
    cv2.imshow("Foreground", fg)
    
    # handle user input
    k = cv2.waitKey(1)

    if k == 0x1b:
      print 'ESC pressed.  Exiting infinite loop.'
      break

  camera.release()
  cv2.destroyAllWindows()

