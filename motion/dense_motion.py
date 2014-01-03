import cv2
import numpy as np
import math

if __name__ == "__main__":

  # open the camera
  camera = cv2.VideoCapture(1)

  _,image_prev = camera.read()
  image_prev = cv2.flip(image_prev, 1)
  sz = image_prev.shape[0:2];
  gray_prev = cv2.cvtColor(image_prev, cv2.cv.CV_RGB2GRAY)

  while True:

    # grab an image from the camera
    _,image = camera.read()
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray_prev,gray,None,
        pyr_scale=.5,
        levels=5,winsize=10,iterations=10,poly_n=5,poly_sigma=1.5,flags=0)
    flow_split = cv2.split(flow)

    # make picture to show user
    vMag, vAngle = cv2.cartToPolar(flow_split[0], flow_split[1])

    vAngle = vAngle * (180 / math.pi)  
    saturation = np.ones(sz, dtype=np.float32)
    vMag = vMag / 10 

    flow_vis = cv2.merge([vAngle, saturation, vMag])
    flow_vis = cv2.cvtColor(flow_vis, cv2.cv.CV_HSV2BGR)

    cv2.imshow("Live Video",image) 
    cv2.imshow("vx", flow_vis)
    
    # handle user input
    k = cv2.waitKey(1)

    if k == 0x1b:
      print 'ESC pressed.  Exiting infinite loop.'
      break
    elif k == ord('0'):
      camera.release()
      camera = cv2.VideoCapture(0)
    elif k == ord('1'):
      camera.release()
      camera = cv2.VideoCapture(1)
    elif k == ord('2'):
      camera.release()
      camera = cv2.VideoCapture(2)
    
    gray_prev, gray = gray, gray_prev
    image_prev = image

  camera.release()
  cv2.destroyAllWindows()

