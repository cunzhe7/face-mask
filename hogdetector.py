import numpy as np
import cv2 as cv
import sys
import dlib


face_detector = dlib.get_frontal_face_detector()
imgNum = sys.argv[1]

img = cv.imread(str(imgNum)+".png")

size = 3
img = cv.resize(img, (img.shape[1] * size, img.shape[0] * size))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces = face_detector(gray, 1)

for (i, rect) in enumerate(faces):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

img = cv.resize(img, (img.shape[1] // size, img.shape[0] // size))
cv.imshow("img", img)
cv.imwrite("hogexample.png",img)
cv.waitKey(0)