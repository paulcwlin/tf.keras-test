from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model_a = Sequential()
model_a.add(Dense(512, activation='relu', input_dim = 784))
model_a.add(Dense(10, activation='softmax'))

model_b = Sequential([
    Dense(512, activation='relu', input_dim = 784),
    Dense(10, activation='softmax')])