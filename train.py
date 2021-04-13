from modelNet import *
import numpy as np
import tensorflow as tf

# from tensorflow.keras import layers
# from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard 
# import pydot
# from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
# from tensorflow.resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
import datetime
# %matplotlib inline

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)




def main():
    #define and compile model.

    model = simpleModel(input_shape=(64,64,3),classes=12)
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    #load the dataset
    x = np.load('./data/x.npy')
    # x = x.reshape((4685,128,128,3))
    y = np.load('./data/y.npy')
    # print(y.shape,x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 400)
    x_train = x_train / 128.0 -1
    x_test = x_test /128.0 -1
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    #set model callback.
    save_weights = ModelCheckpoint("./models/model.h5", 
                                   save_best_only=True, monitor='val_acc')
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [save_weights, tensorboard_callback]

    #train the model.
    model.fit(x_train, y_train, epochs = 100, batch_size=32, 
              validation_data = (x_test, y_test), callbacks=callbacks)

if __name__ == '__main__':
    main()
    