import cv2

# Set the path for the cascade file and set the camera index. Camera index
# is one on my laptop because the 0th camera accesses the preloaded camera
# software instead of the actual camera.  Index should be 0 for most other
# computers
HAAR_CASCADE_PATH = "C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
CAMERA_INDEX = 1

# Defines the function to detect faces using the cascade file and an input
# image from the camera
def detect_faces(image):
    faces = []
    detected = cv2.cv.HaarDetectObjects(image, cascade, storage, 1.2, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (100,100))
    if detected:
        for (x,y,w,h),n in detected:
            faces.append((x,y,w,h))
    return faces

# Main function of the program
if __name__ == "__main__":

    # Loads the cascade file and sets up to capture images from the camera
    capture = cv2.cv.CaptureFromCAM(CAMERA_INDEX)
    storage = cv2.cv.CreateMemStorage()
    cascade = cv2.cv.Load(HAAR_CASCADE_PATH)
    faces = []

    acc = None

    while True:
        image = cv2.cv.QueryFrame(capture)
        cv2.cv.Flip(image, None, 1)
        
        faces = detect_faces(image)
     
        resizedFace = cv2.cv.CreateImage((250, 250), image.depth, image.nChannels)
     
        for (x,y,w,h) in faces:
            cv2.cv.Rectangle(image, (x,y), (x+w,y+h), 255)
            
            extractedFace = image[y:y+h, x:x+w]
            cv2.cv.Resize(extractedFace, resizedFace)
        
            if acc is None:
                acc = cv2.cv.CreateImage(cv2.cv.GetSize(resizedFace), 32, 3)
                accShow = cv2.cv.CreateImage(cv2.cv.GetSize(resizedFace), 8, 3)

            cv2.cv.RunningAvg(resizedFace, acc, .01, None)
            cv2.cv.ConvertScaleAbs(acc, accShow)
        
            cv2.cv.ShowImage("Extracted Face", resizedFace)
            cv2.cv.ShowImage("Running Average", accShow)
        
        cv2.cv.ShowImage("Face Detection", image)
           
        k = cv2.waitKey(1)

        if k == 0x1b:
            print 'ESC pressed.  Exiting infinite loop.'
            break
        
cv2.cv.DestroyAllWindows()
