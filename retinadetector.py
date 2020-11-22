import numpy as np
import cv2 as cv
import sys
import face_detection


detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)


imgNum = sys.argv[1]


origimg = cv.imread(str(imgNum)+".png")
img = cv.cvtColor(origimg, cv.COLOR_BGR2RGB)

detection = detector.detect(img)

faces = []

for i in range(detection.shape[0]):
    xmin, ymin, xmax, ymax = detection[i,:4].astype("int")

    img =  cv.rectangle(origimg, (xmin,ymin), (xmax, ymax), (0,255,0), 2)


cv.imshow('img', origimg)
# cv.imwrite('haaraltex1.png',img)
cv.waitKey(0)
cv.destroyAllWindows()