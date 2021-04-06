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
for annotation in annotations:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]
    image_path = images[image_id]["file_name"]
    print(image_path)

