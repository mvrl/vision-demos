import cv2
import sys

# Detect a face in an image
# (started with
# http://japskua.wordpress.com/2010/08/04/detecting-eyes-with-python-opencv/)

haar_scale = 1.1
min_neighbors = 2
haar_flags = 0 
min_size = (20,20)

if len(sys.argv) == 1:
  image = cv.LoadImage("images/judybat.jpg")
else:
  image = cv.LoadImage(sys.argv[1])

faceCascade = cv.Load("haar/face.xml")

# Allocate the temporary images
gray = cv.CreateImage((image.width, image.height), 8, 1)

# Convert color input image to grayscale
cv.CvtColor(image, gray, cv.CV_BGR2GRAY)

faces = cv.HaarDetectObjects(gray, faceCascade, cv.CreateMemStorage(0),
    haar_scale, min_neighbors, haar_flags, min_size)

if faces:

  for ((x, y, w, h), n) in faces:
    # the input to cv.HaarDetectObjects was resized, so scale the
    # bounding box of each face and convert it to two CvPoints
    pt1 = (int(x), int(y))
    pt2 = (int((x + w)), int((y + h)))
    cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)

cv.NamedWindow("Face")
cv.ShowImage("Face", image)
cv.WaitKey(0)
cv.DestroyWindow("Face")

