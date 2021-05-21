from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

cnn1 = Sequential()
cnn2 = Sequential()

cnn1.add(Conv2D(filters=32,
               kernel_size=[3,3],
               padding='same',
               input_shape=[100,100,1]))

cnn1.add(MaxPooling2D(pool_size=[2,2],
                     strides=2))

cnn2.add(Conv2D(filters=32,
                kernel_size=[3,3],
                padding='same',
                input_shape=[100,100,1]))

cnn2.add(AveragePooling2D(pool_size=[2,2],
                          strides=2))
cnn1.summary()
cnn2.summary()