import cv2
import numpy as np
import math
import time

#
# A simple example showing how to match almost simultaneous images from two webcams using SURF
#
# Possible next steps:
#   - solve for fundamental matrix
#   - visualize matches
#   - filter bad matches

if __name__ == "__main__":

  # open the camera
  c1 = cv2.VideoCapture(0)
  c2 = cv2.VideoCapture(1)

  surf = cv2.SURF()

  while True:

    _,im1 = c1.read()
    _,im2 = c2.read()

    im1g = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY);
    im2g = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY);

    kp1,desc1 = surf.detect(im1g,None,useProvidedKeypoints=False);
    kp2,desc2 = surf.detect(im2g,None,useProvidedKeypoints=False);

    desc1 = desc1.reshape(len(kp1),-1)
    desc2 = desc2.reshape(len(kp2),-1)

    samples = np.array(desc1);
    responses = np.arange(len(kp1),dtype=np.float32)

    knn = cv2.KNearest()
    knn.train(samples,responses)

    for h,des in enumerate(desc2):
       des = np.array(des,np.float32).reshape((1,128))
       retval, results, neigh_resp, dists = knn.find_nearest(des,1)
       res,dist =  int(results[0][0]),dists[0][0]

       if dist<0.1: # draw matched keypoints in red color
         color = (0,0,255)
       else:  # draw unmatched in blue color
         color = (255,0,0)

       #Draw matched key points on original image
       x,y = kp1[res].pt
       center = (int(x),int(y))
       cv2.circle(im1,center,2,color,-1)

       #Draw matched key points on template image
       x,y = kp2[h].pt
       center = (int(x),int(y))
       cv2.circle(im2,center,2,color,-1)

    cv2.imshow('im1',im1)
    cv2.imshow('im2',im2)
    
    # handle user input
    k = cv2.waitKey(1)

    if k == 0x1b:
      print 'ESC pressed.  Exiting infinite loop.'
      break

  c1.release()
  c2.release()
  cv2.destroyAllWindows()

