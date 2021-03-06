# %matplotlib inline
# import tensorflow as tf
import numpy as np
import cv2 as cv
# import matplotlib.pylab as plt
# import pandas as pd
import os

datasetPath = "./Dataset"

#导入数据
paths = os.listdir(datasetPath)
imgs = []
labels = []
for index,path in  enumerate(paths):
    label = index
    files = os.listdir(datasetPath + '/' + path)
    for imgfile in files:
        img = cv.imread(datasetPath + '/'+path+'/'+imgfile)
        img = cv.resize(img,(64,64))
        # cv.imshow('img',img)
        # cv.waitKey()
        imgs.append(img)
        labels.append(label)
        print(datasetPath + '/'+path+'/'+imgfile+':'+str(label))
np.save(datasetPath + '/' + 'x.npy',imgs)
np.save(datasetPath + '/' + 'y.npy',labels)

