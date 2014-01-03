import cv
import sys

if __name__ == "__main__":

  # open the camera
  capture = cv.CaptureFromCAM(1)

  if not capture:
    print "Error opening capture device"
    sys.exit(1)

  acc = None

  while True:

    # grab an image from the camera
    image = cv.QueryFrame(capture)

    # flip it left right because that is what people are used to
    cv.Flip(image, None, 1)

    if acc is None:
      acc = cv.CreateImage(cv.GetSize(image), 32, 3)
      accShow = cv.CreateImage(cv.GetSize(image), 8, 3)

    cv.RunningAvg(image, acc, .01, None)
    cv.ConvertScaleAbs(acc, accShow)

    cv.ShowImage("Live Video",image) 
    cv.ShowImage("Blurry Video", accShow)
    
    # handle user input
    k = cv.WaitKey(1)

    if k == 0x1b:
      print 'ESC pressed.  Exiting infinite loop.'
      break

  cv.DestroyAllWindows()

