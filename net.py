import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.utils import to_categorical 

import tensorboard

# import numpy as np
# import matplotlib as plt
def genVGG(input_shape=(128,128,1)):
    deep_model = tf.keras.Sequential()
    
    # BLOCK 1
    deep_model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1', input_shape = input_shape ))   
    deep_model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2'))
    deep_model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block1_pool'))
    
    # BLOCK2
    deep_model.add(keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1'))   
    deep_model.add(keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2'))
    deep_model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block2_pool'))
    
    # BLOCK3
    deep_model.add(keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1'))   
    deep_model.add(keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2'))
    deep_model.add(keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3'))
    deep_model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block3_pool'))
    
    # BLOCK4
    deep_model.add(keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1'))   
    deep_model.add(keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2'))
    deep_model.add(keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3'))
    deep_model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block4_pool'))
    
    # BLOCK5
    deep_model.add(keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv1'))   
    deep_model.add(keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2'))
    deep_model.add(keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3'))
    deep_model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'block5_pool'))
    
    deep_model.add(keras.layers.Flatten())
    deep_model.add(keras.layers.Dense(4096, activation = 'relu', name = 'fc1'))
    deep_model.add(keras.layers.Dropout(0.5))
    deep_model.add(keras.layers.Dense(4096, activation = 'relu', name = 'fc2'))
    deep_model.add(keras.layers.Dropout(0.5))
    deep_model.add(keras.layers.Dense(60, activation = 'softmax', name = 'prediction'))

    # ??????????????????
    # deep_model.summary()

    return deep_model

if __name__ == '__main__':
    model = genVGG()
    model.summary()
    x = np.load('./data/x.npy')
    x = x.reshape((4685,128,128,1))
    y = np.load('./data/y.npy')
    # print(y.shape,x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 400)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    opt = Adam(lr=0.000001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
    # early_stop = EarlyStopping(patience=20)
    # reduce_lr = ReduceLROnPlateau(patience=15)
    save_weights = ModelCheckpoint("./models/model.h5", 
                                   save_best_only=True, monitor='val_acc')
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [save_weights, tensorboard_callback]
    model.fit(x_train, y_train, epochs = 100, batch_size=32, 
              validation_data = (x_test, y_test), callbacks=callbacks)