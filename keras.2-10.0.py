import pandas as pd
import numpy as np
import util2 as u
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv(r'C:\Users\d19fd\Documents\tf.Keras\Sample code\ch02\Admission_Predict_Ver1.1.csv')

rnd = np.random.RandomState(6)
ds = df.values
rnd.shuffle(ds)

x = ds[:, 1:8]
y = ds[:, 8]

x_train = x[:400]
y_train = y[:400]

x_test = x[400:]
y_test = y[400:]

mean = x_train.mean(axis=0) 
std = x_train.std(axis=0) #Standar Deviation

x_train -= mean
x_train /= std

x_test -= mean
x_test /= std

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=7))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

wt = model.get_weights()
ksize = len(x_train) //4
all_his_mae = []
all_his_val = []

for i in range(4):
    print(f'The {i} fold training and validation', end='')
    fr, to = i*ksize, (i+1)*ksize
    
    x_val = x_train[fr : to]
    x_trn = np.concatenate([x_train[ : fr],
                            x_train[to : ]], axis=0)
    y_val = y_train[fr : to]
    y_trn = np.concatenate([y_train[ : fr],
                            y_train[to : ]], axis=0)
    
    model.set_weights(wt)
    history = model.fit(x_trn, y_trn,
                        validation_data=(x_val, y_val),
                        epochs=200,
                        batch_size=4,
                        verbose=2) #0:不顯示 1:顯示進度條= 2:不顯示進度條
    hv = history.history['val_mae']
    idx = np.argmin(hv)
    val = hv[idx]
    
    u.plot(history.history,
           ('mae', 'val_mae'),
           f'k={i} Best val_mae at epoch={idx+1} val_mae={val:.3f}',
           ('Epoch', 'mae'), ylim=(0.03, 0.08))
    
    all_his_mae.append(history.history['mae'])
    all_his_val.append(history.history['val_mae'])
    
avg_mae = np.mean(all_his_mae, axis=0)
avg_val = np.mean(all_his_val, axis=0)
idx = np.argmin(avg_val)
val = avg_val[idx]

print(f'Best mean period={idx+1}, val_mae={val:.3f}')

u.plot({'avg_mae': avg_mae, 'avg_val_mae': avg_val},
       ('avg_mae', 'avg_val_mae',),
       f'Best avg val_mae at epoch {idx+1} val_mae={val:.3f}',
       ('Epoch', 'mae'), ylim=(0.03, 0.08))
       
print(f'用所有的訓練資料重新訓練到第 {idx+1} 週期')
model.set_weights(wt)
history = model.fit(x_train, y_train,
                    epochs=idx+1,
                    batch_size=4,
                    verbose=0)

loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f'用測試資料評估結果 mae={mae:.3f}') 