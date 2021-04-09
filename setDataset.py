# %matplotlib inline
# import tensorflow as tf
import numpy as np
import cv2 as cv
# import matplotlib.pylab as plt
# import pandas as pd
import json

def formatSegmentList(Segmnetlist):
    segmentation = []
    for item in Segmnetlist:
        newList = []
        for i in range(0,len(item),2):
            dot = []
            dot.append(item.pop(0))
            dot.append(item.pop(0))
            newList.append(dot)
        newList = np.array(newList,np.int32)
        newList = newList.reshape(-1,1,2)
        segmentation.append(newList)
    return segmentation
        
def img_paste(image, polygon_list):
    # print(polygon_list)
    #创建一个和原图一样的全0数组
    im = np.zeros(image.shape[:2], dtype="uint8")
    #把所有的点画出来
    cv.polylines(im, polygon_list, True, 255)
    #把所有点连接起来，形成封闭区域
    cv.fillPoly(im, polygon_list, 255)
    mask = im
    #将连接起来的区域对应的数组和原图对应位置按位相与
    masked = cv.bitwise_and(image, image, mask=mask)
    #cv2中的图片是按照bgr顺序生成的，我们需要按照rgb格式生成
    b,g,r = cv.split(masked)
    masked = cv.merge([r, g, b])
    return masked


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
    segmentation = annotation["segmentation"]
    try:
        img = cv.imread(image_path)
        img = img_paste(img,formatSegmentList(segmentation))
        if rect[3] > rect[2]:
            rect[0] = rect[0] + rect[2]/2 - rect[3]/2
            rect[2] = rect[3]
        elif rect[3] < rect[2]:
            rect[1] = rect[1] + rect[3]/2 - rect[2]/2
            rect[3] = rect[2] 
        roi = img[int(rect[1]):int(rect[1])+int(rect[3]),int(rect[0]):int(rect[0])+int(rect[2])]
        roi = cv.resize(roi,(128,128))
        print("正在处理图片"+str(image_id)+",标签："+str(category_id))
    except Exception as Error:
        print("图片"+str(image_id)+"处理失败")
        print(Error)
    else:
        imgs.append(roi)
        labels.append(category_id)
    #     cv.imshow('img',roi)
    # cv.waitKey()
np.save('./data/x.npy',imgs)
np.save('./data/y.npy',labels)

