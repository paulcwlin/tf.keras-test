from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

model = Sequential([
    Dense(512, input_dim = 784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')])

model.summary()
