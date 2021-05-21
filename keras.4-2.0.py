from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

data = [0,1,1,1,1,1,1,2,2,2,2,3,3] * 5
data = to_categorical(data)

data_gen = TimeseriesGenerator(data,
                               data,
                               length=1,
                               batch_size=1)

#print(data_gen[0])

model = Sequential()
model.add(layers.SimpleRNN(10,
                          stateful=True,
                          batch_input_shape=(1, None, 4)))

model.add(layers.Dense(4,
                       activation='softmax'))
#model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

epochs = 50

for i in range(epochs):
    print('Epoch', i+1, '/', epochs)
    model.fit_generator(data_gen,
                        epochs=1,
                        shuffle=False)
    model.reset_states()

