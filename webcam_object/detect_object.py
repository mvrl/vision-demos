# A python program that detects objects in your webcam
# Author: Dr. Nathan Jacobs (http://cs.uky.edu/~jacobs/)

# load the computer vision library (http://opencv.willowgarage.com/)
import cv

# define a function that finds all the objects in the image
def detect(image, objectDescription):

  # define the parameters
  haar_scale = 1.1
  min_neighbors = 6
  haar_flags = 0
  min_size = (40,40)

  objects = cv.HaarDetectObjects(image, objectDescription, cv.CreateMemStorage(0),
      haar_scale, min_neighbors, haar_flags, min_size)

  return objects 

if __name__ == "__main__":

  # Detect an object in a a live webcam stream 

  # This part of the program that only runs when you call it from the
  # command line.  For example, "python detect_face.py".

  print "Select the window and press ESC to quit."

  # load up the description of the object detector
  objectDescription = cv.Load("haar/face.xml")

  # use different object detectors
  # objectDescription = cv.Load("haar/haarcascade_mcs_nose.xml")
  # objectDescription = cv.Load("haar/haarcascade_mcs_righteye.xml")
  # objectDescription = cv.Load("haar/haarcascade_mcs_upperbody.xml")
  # objectDescription = cv.Load("haar/haarcascade_frontalface_default.xml")

  # open the camera
  capture = cv.CreateCameraCapture(0)

  cv.NamedWindow("Object Detection", cv.CV_WINDOW_NORMAL)

  while True:

    # grab an image from the camera
    image = cv.QueryFrame(capture)

    # flip it left right (otherwise it looks strange)
    cv.Flip(image, None, 1)

    # detect all the objects 
    objects = detect(image, objectDescription)

    # TODO: sort by number of detections (so good ones are on top)
    # TODO: make a good color coding for good faces and bad faces
    # TODO: possibly display another window that computes the average
    # of all the faces in the field of view

    # draw the objects in the image
    for ((x, y, w, h), n) in objects:
      cv.Rectangle(image, (x,y), (x+w,y+h), cv.RGB(n*3, 0, 0), 3, 8, 0)

    # show the image
    cv.ShowImage("Object Detection", image)
    
    # get any user input (wait for 1 millisecond)
    k = cv.WaitKey(1)

    if k == 0x1b:
      print 'ESC pressed.  Ending program.'
      break

  cv.DestroyAllWindows()

