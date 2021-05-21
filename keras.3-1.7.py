from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

cnn = Sequential()

cnn.add(Conv2D(1,
               [3,3],
               padding='same',
               input_shape=[28,28,1]))

cnn.summary()