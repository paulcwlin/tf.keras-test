import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import util4 as u
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite as iter_inf

def statefulGenerator(data,
                      target,
                      length=1,
                      sampling_rate=1,
                      stride=1,
                      start_index=0,
                      end_index=None,
                      batch_size=128):
    if end_index == None:
        end_index = len(data)-1
    data_len = ((end_index - start_index + 1)-length)//stride
    tmp_batch_size = data_len//batch_size
    
    end_index = tmp_batch_size * batch_size * stride + length -1 + start_index
    
    data_gen = TimeseriesGenerator(data,
                                   target,
                                   length=length,
                                   sampling_rate=sampling_rate,
                                   stride=stride,
                                   start_index=start_index,
                                   end_index=end_index,
                                   batch_size=batch_size)
    new_data = []
    new_target = []
    
    for i in data_gen:
        new_data.append(i[0])
        new_target.append(i[1])
        
    new_data = np.array(new_data)
    new_target = np.array(new_target)
    
    new_data = new_data.transpose(1,0,2,3)
    new_target = new_target.transpose(1,0,2)
    
    new_data_gen = []
    
    for i in range(len(new_data)):
        new_data_gen.append((new_data[i], new_target[i]))
        
    return new_data_gen


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
state_train_gen = statefulGenerator(data,
                                target,
                                length = length,
                                sampling_rate = sampling_rate,
                                stride = stride,
                                start_index = 0,
                                end_index = 100000,
                                batch_size = batch_size)
# Generate validation data
state_val_gen = statefulGenerator(data,
                              target,
                              length = length,
                              sampling_rate = sampling_rate,
                              stride = stride,
                              start_index = 100001,
                              end_index = 130000,
                              batch_size = batch_size)
# Generate Test data
state_test_gen = statefulGenerator(data,
                               target,
                               length = length,
                               sampling_rate = sampling_rate,
                               stride = stride,
                               start_index = 130001,
                               end_index = None,
                               batch_size = batch_size)

#print(train_gen[0][0].shape)

# Used stateful Recurrent Neural Network
state_model = Sequential()
state_model.add(layers.SimpleRNN(10,
                                 stateful=True,
                                 batch_input_shape=(32,12,1)))
state_model.add(layers.Dense(10, activation='relu'))
state_model.add(layers.Dense(1))
#state_model.summary()

state_model.compile(optimizer='rmsprop',
                    loss='mse',
                    metrics=['mae'])

epochs = 50
loss = []
val_loss = []

for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    state_history = state_model.fit(iter_inf(state_train_gen),
                                    steps_per_epoch = len(state_train_gen),
                                    validation_data = iter_inf(state_val_gen),
                                    validation_steps = len(state_val_gen),
                                    epochs = 1,
                                    verbose = 1,
                                    shuffle = False)
    
    loss.append(state_history.history['loss'])
    val_loss.append(state_history.history['val_loss'])
    state_model.reset_state()
    
print('stateRNN平均溫度誤差 : ', state_history.history['val_mae'][-1]*std)

u.plot({'loss' : loss, 'val_loss' : val_loss},
       ['loss', 'val_loss'],
       title='Training & Validation loss',
       xyLabel=['Epoch', 'Loss'])