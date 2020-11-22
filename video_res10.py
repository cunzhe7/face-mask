import numpy as np
import cv2 as cv
import sys
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
from imutils.video import VideoStream

prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"

face_detector = cv.dnn.readNet(prototxtPath,weightsPath)
mask_classifier = load_model("best.h5")

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()
length = 600
while True:
    ret, img = cam.read()

    if not ret:
        print("Can't receive stream")
        break
    img = cv.resize(img, (length,length))

    blob = cv.dnn.blobFromImage(img, 1.0, (length,length), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    locs = [] 
    for i in range(detections.shape[2]):
        score = detections[0,0,i,2]

        if score > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([length, length, length, length])
            l, t, r, d = box.astype("int")
            l, t = max(0,l), max(0,t)
            r,d = min(length-1, r), min(length-1, d)
            face = img[t:d,l:r]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (100,100))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((l, t,r, d))

    preds = []
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_classifier.predict(faces, batch_size=32)

    for (box, pred) in zip(locs, preds):
        x, y, w, h = box
        w -= x
        h -= y

        label = "Mask" if pred > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv.putText(img, label, (x, y - 10) , cv.FONT_HERSHEY_PLAIN, 1, color,2)
        img = cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
        
        img = cv.resize(img, (img.shape[1] , img.shape[0] ))
    cv.imshow('frame', img)
    if cv.waitKey(10) == ord('q'):
        break
# cv.imwrite('pred_by_image_eg1.png',img)
# cam.release()
cam.release()
cv.destroyAllWindows()