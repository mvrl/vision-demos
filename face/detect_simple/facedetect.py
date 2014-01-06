import cv2
 
HAAR_CASCADE_PATH = "C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml"
CAMERA_INDEX = 1
 
def detect_faces(image):
    faces = []
    detected = cv2.cv.HaarDetectObjects(image, cascade, storage, 1.2, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (100,100))
    if detected:
        for (x,y,w,h),n in detected:
            faces.append((x,y,w,h))
    return faces
 
if __name__ == "__main__":

    capture = cv2.cv.CaptureFromCAM(CAMERA_INDEX)
    storage = cv2.cv.CreateMemStorage()
    cascade = cv2.cv.Load(HAAR_CASCADE_PATH)
    faces = []
 
    i = 0
    while True:
        image = cv2.cv.QueryFrame(capture)

        cv2.cv.Flip(image, None, 1)        

        faces = detect_faces(image)
 
        for (x,y,w,h) in faces:
            cv2.cv.Rectangle(image, (x,y), (x+w,y+h), 255)
 
        cv2.cv.ShowImage("Face Detection", image)
        i += 1
        cv2.waitKey(1)
