import numpy as np
import cv2 as cv
import sys
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
net = load_model("best.h5")

cam = cv.VideoCapture(0)

size = 2

if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cam.read()

    if not ret:
        print("Can't receive stream")
        break

    img = cv.flip(img, 1,1)
    img = cv.resize(img, (img.shape[1] * size, img.shape[0] * size))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)

    for (x,y,w,h) in faces:
    
        face = img[y:y+h,x:x+w]
        face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        face = cv.resize(face, (100,100))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face,axis=0)

        sth = net.predict(face)
        label = "Mask" if sth > 0.5 else "No Mask"
        color = (0,255,0) if sth > 0.5 else (0,0,255)

        cv.putText(img, label, (x, y - 10) , cv.FONT_HERSHEY_PLAIN, 1, color,2)
        img = cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
        
    img = cv.resize(img, (img.shape[1] // size, img.shape[0] // size))
    cv.imshow('frame', img)
    if cv.waitKey(10) == ord('q'):
        break
# cv.imwrite('pred_by_image_eg1.png',img)
cam.release()
cv.destroyAllWindows()