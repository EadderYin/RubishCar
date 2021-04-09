import cv2 as cv
import numpy as np
import os

x = np.load('./data/x.npy')
y = np.load('./data/y.npy')

for i in range(0,len(x)):
    print(str(i)+':'+str(y[i]))
    # cv.imshow('IMG',x[i])
    if not cv.imwrite('./data/img/'+str(y[i])+'/'+str(i)+'.bmp',x[i]):
        os.mkdir('./data/img/'+str(y[i]))
        cv.imwrite('./data/img/'+str(y[i])+'/'+str(i)+'.bmp',x[i])
    # if cv.waitKey() == 13:
        # break