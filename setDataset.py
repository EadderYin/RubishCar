# %matplotlib inline
# import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import json
import 

#导入数据
URL = './data/annotations.json'
with open(URL,"r") as f:
    dataframe = json.load(f)
imgrects = dataframe["annotations"]
for imgrect in imgrects:
    # print(str(imgrect["bbox"])+'\n')
    if bbox is None:
        bbox = tf.constant(imgrect["bbox"],dtype=tf.float32,shape=[1,1,4])


# dataframe = pd.read_csv(URL)
# dataframe.head()

