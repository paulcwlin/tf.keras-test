from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
import util3 as u
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train[0][0][0])

x_train_norm = x_train.astype('float32')/255
x_test_norm = x_test.astype('float32')/255

#print(x_train_norm[0][0][0])

#print(y_train[0])

y_train_onehot = utils.to_categorical(y_train, 10)
y_test_onehot = utils.to_categorical(y_test, 10)

#print(y_train_onehot[0])

cnn = Sequential()

cnn.add(Conv2D(filters=32,
               kernel_size=[3,3],
               activation='relu',
               padding='same',
               input_shape=[32,32,3]))

cnn.add(Dropout(0.25))
cnn.add(MaxPooling2D(pool_size=[2,2]))

cnn.add(Conv2D(filters=64,
               kernel_size=[3,3],
               padding='same',
               activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(MaxPooling2D(pool_size=[2,2]))

cnn.add(Flatten())
cnn.add(Dropout(0.25))

cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.25))

cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics='acc')

history = cnn.fit(x=x_train_norm,
                  y=y_train_onehot,
                  batch_size=128,
                  epochs=20,
                  validation_split=0.1)

u.plot(history.history,
       ('acc', 'val_acc'),
       'Training & Validation Acc',
       ('Epoch', 'Acc'))

u.plot(history.history,
       ('loss', 'val_loss'),
       'Training & Validation Loss',
       ('Epoch', 'Loss'))

cnn.save('cifar10model.h5')

old_cnn = load_model(r'C:\Users\d19fd\Documents\tf.Keras\model\cifar10model.h5')

test_loss, test_val = cnn.evaluate(x_test_norm, y_test_onehot)

print('測試資料損失值 :', test_loss)
print('測試資料準確率 :', test_val)