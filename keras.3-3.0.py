from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

cnn = Sequential()

cnn.add(Conv2D(filters=8,
               kernel_size=[3,3],
               padding='same',
               input_shape=[100,100,1]))

cnn.add(MaxPooling2D(pool_size=[2,2],
                     strides=2))

cnn.add(Flatten())

cnn.summary()