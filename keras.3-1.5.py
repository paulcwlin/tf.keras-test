from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn1 = Sequential()
cnn2 = Sequential()
cnn3 = Sequential()

cnn1.add(Conv2D(filters=32,
               kernel_size=(3,3),
               input_shape=(5,5,1)))

cnn2.add(Conv2D(filters=32,
                kernel_size=(3,3),
                use_bias=False,
                input_shape=(5,5,1)))

cnn3.add(Conv2D(filters=1,
                kernel_size=[3,3],
                strides=[2,1],
                input_shape=[28,28,1]))


cnn1.summary()
cnn2.summary()
cnn3.summary()