import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


sample = np.array([[[0], [1]],
                   [[1], [1]],
                   [[1], [2]]])

label = np.array([1, 2, 0])

sample = to_categorical(sample)
print(sample)

label = to_categorical(label)

model = Sequential()

model.add(layers.SimpleRNN(10,
                           input_shape=(2, 3)))
model.add(layers.Dense(3,
                       activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(sample,
                    label,
                    epochs=100)

predict = model.predict_classes(sample)
print(predict)
