import numpy as np
import cv2 as cv
import sys
import dlib



face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
imgNum = sys.argv[1]


img = cv.imread(str(imgNum)+".png")

size = 3
img = cv.resize(img, (img.shape[1] * size, img.shape[0] * size))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray)

for (x,y,w,h) in faces:
    img = cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
img = cv.resize(img, (img.shape[1] // size, img.shape[0] // size))
cv.imshow('img', img)
# cv.imwrite('haaraltex1.png',img)
cv.waitKey(0)
cv.destroyAllWindows()