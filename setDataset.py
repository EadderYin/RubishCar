# %matplotlib inline
# import tensorflow as tf
import numpy as np
import cv2 as cv
# import matplotlib.pylab as plt
# import pandas as pd
import json

#导入数据
URL = './data/annotations.json'
with open(URL,"r") as f:
    dataframe = json.load(f)
annotations = dataframe["annotations"]
images = dataframe["images"]
imgs = []
labels = []
for annotation in annotations:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    image_path = "data/" + images[image_id]["file_name"]
    rect = annotation["bbox"]
    try:
        img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
        if rect[3] > rect[2]:
            rect[0] = rect[0] + rect[2]/2 - rect[3]/2
            rect[2] = rect[3]
        elif rect[3] < rect[2]:
            rect[1] = rect[1] + rect[3]/2 - rect[2]/2
            rect[3] = rect[2] 
        roi = img[int(rect[1]):int(rect[1])+int(rect[3]),int(rect[0]):int(rect[0])+int(rect[2])]
        roi = cv.resize(roi,(128,128))
        print("正在处理图片"+str(image_id)+",标签："+str(category_id))
    except:
        print("图片"+str(image_id)+"处理失败")
    else:
        imgs.append(roi)
        labels.append(category_id)
np.save('./data/x.npy',imgs)
np.save('./data/y.npy',labels)

