# %matplotlib inline
# import tensorflow as tf
import numpy as np
import cv2 as cv
# import matplotlib.pylab as plt
# import pandas as pd
import os

#导入数据
paths = os.listdir('./data/new_img')
imgs = []
labels = []
for path in paths:
    label = int(path.split('_')[0])
    files = os.listdir('./data/new_img/'+path)
    for imgfile in files:
        img = cv.imread('./data/new_img/'+path+'/'+imgfile)
        img = cv.resize(img,(128,128))
        # cv.imshow('img',img)
        # cv.waitKey()
        imgs.append(img)
        labels.append(label)
        print('./data/new_img/'+path+'/'+imgfile+':'+str(label))
np.save('./data/x.npy',imgs)
np.save('./data/y.npy',labels)

