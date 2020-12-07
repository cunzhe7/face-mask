import numpy as np
import cv2 as cv
import sys


prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv.dnn.readNet(prototxtPath, weightsPath)

imgNum = sys.argv[1]
img = cv.imread(str(imgNum)+".png")

h, w = img.shape[:2]

blob = cv.dnn.blobFromImage(img, 1.0, (300,300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0,0,i,2 ]

    if confidence > float(sys.argv[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        cv.rectangle(img, (startX, startY), (endX, endY), (0,255,0), 2)

cv.imshow('img', img)
cv.waitKey(0)