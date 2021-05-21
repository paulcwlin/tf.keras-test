from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras import layers

dinner = [0,1,1,2,0,1,1,2]
dinner = to_categorical(dinner)
#print(dinner)

data_gen = TimeseriesGenerator(dinner,
                               dinner,
                               length=2,
                               sampling_rate=1,
                               stride=1,
                               batch_size=2)

#print(data_gen[0])

model = Sequential()
model.add(layers.SimpleRNN(10,
                           input_shape=(2, 3)))
model.add(layers.Dense(3,
                       activation='softmax'))

#model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

train_history = model.fit(data_gen,
                          epochs=100)

prediction = model.predict_classes(data_gen)
print(prediction)