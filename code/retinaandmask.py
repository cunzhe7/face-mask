import numpy as np
import cv2 as cv
import sys
import face_detection
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


detector = face_detection.build_detector(
  "RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

mask_classifier = load_model("best.h5")

imgNum = sys.argv[1]


origimg = cv.imread(str(imgNum)+".png")
w, h = origimg.shape[:2]
if w < 1000 and h < 1000:
    origimg = cv.resize(origimg, (h*2, w*2))
img = cv.cvtColor(origimg, cv.COLOR_BGR2RGB)

detection = detector.detect(img)


faces = []
locs = []
for i in range(detection.shape[0]):
    xmin, ymin, xmax, ymax = detection[i,:4].astype("int")
    face = origimg[ymin:ymax,xmin:xmax]
    face = cv.resize(face, (100,100))
    face = img_to_array(face)
    face = preprocess_input(face)
    faces.append(face)
    locs.append((xmin, ymin, xmax, ymax))
    
preds = []
if len(faces) > 0:
    faces = np.array(faces, dtype="float32")
    preds = mask_classifier.predict(faces, batch_size=32)
    
for (box, pred) in zip(locs, preds):
    xmin, ymin, xmax, ymax = box
    if pred > 0.5:
        img =  cv.rectangle(origimg, (xmin,ymin), (xmax, ymax), (0,255,0), 2)
    else:
        img =  cv.rectangle(origimg, (xmin,ymin), (xmax, ymax), (0,0,255), 2)

cv.imshow('img', origimg)
cv.imwrite('retinamobileex.png',img)
cv.waitKey(0)
cv.destroyAllWindows()