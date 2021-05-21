import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import util4 as u


fname = r'C:\Users\d19fd\Documents\tf.Keras Book Sample Code\Sample code\ch04\TY_climate_2015_2018.csv'
f = open(fname, encoding='cp950')
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
del lines[0]

#print(len(lines))
#print(header)

raw_data = []
for i, line in enumerate(lines):
    value = float(line.split(',')[8])
    raw_data.append([value])
    
raw_data = np.array(raw_data)
#plt.plot(raw_data)
#plt.show()

mean = raw_data[:100000].mean()
raw_data -= mean
std = raw_data[:100000].std()
raw_data /= std

#plt.plot(raw_data)
#plt.show()

length = 36
delay = 72
sampling_rate = 3
stride = 36
batch_size = 32

data = raw_data[ : -(delay-1)]
target = raw_data[(delay-1) : ]

# Generate training data
train_gen = TimeseriesGenerator(data,
                                target,
                                length = length,
                                sampling_rate = sampling_rate,
                                stride = stride,
                                start_index = 0,
                                end_index = 100000,
                                batch_size = batch_size)
# Generate validation data
val_gen = TimeseriesGenerator(data,
                              target,
                              length = length,
                              sampling_rate = sampling_rate,
                              stride = stride,
                              start_index = 100001,
                              end_index = 130000,
                              batch_size = batch_size)
# Generate Test data
test_gen = TimeseriesGenerator(data,
                               target,
                               length = length,
                               sampling_rate = sampling_rate,
                               stride = stride,
                               start_index = 130001,
                               end_index = None,
                               batch_size = batch_size)

#print(train_gen[0][0].shape)


# Used Dense Neural Network
dense_model = Sequential()
dense_model.add(layers.Flatten(input_shape=(12, 1)))
dense_model.add(layers.Dense(10, activation='relu'))
dense_model.add(layers.Dense(10, activation='relu'))
dense_model.add(layers.Dense(1))
#dense_model.summary()

dense_model.compile(optimizer='rmsprop',
                    loss='mse',
                    metrics=['mae'])

dense_history = dense_model.fit(train_gen,
                                epochs=100,
                                validation_data=val_gen)

print('DNN平均溫度誤差 :',
      dense_history.history['val_mae'][-1]*std)


val_temp = []

for datas in val_gen:
    for temp in datas[1]:
        val_temp.append(temp)
        
prediction = dense_model.predict(val_gen)

        
u.plot({'val_temp' : val_temp, 'pred_temp' : prediction},
       ['val_temp', 'pred_temp'],
       title='Dense NN',
       xyLabel=['Time', 'Temp'])

plt.plot(dense_history.history['loss'], label='training loss')
plt.plot(dense_history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

