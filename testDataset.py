import cv2 as cv
import numpy as np

x = np.load('./data/x.npy')
y = np.load('./data/y.npy')

for i in range(0,len(x)):
    print(y[i])
    cv.imshow('IMG',x[i])
    if cv.waitKey() == 13:
        break