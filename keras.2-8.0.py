from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import util2 as u

(a_train, b_train), (a_test, b_test) = imdb.load_data(num_words=10000)

tok = Tokenizer(num_words=10000)
x_train = tok.sequences_to_matrix(a_train)
x_test = tok.sequences_to_matrix(a_test)

y_train = b_train.astype('float32')
y_test = b_test.astype('float32')

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=10000))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics='acc')

history = model.fit(x_train, y_train,
                    batch_size=512,
                    epochs=10,
                    validation_split=0.2)

u.plot(history.history,
       ('loss', 'val_loss'),
       'Training & Validation Loss',
       ('Epoch', 'Loss'))
u.plot(history.history,
       ('acc', 'val_acc'),
       'Training & Validation Acc',
       ('Epoch', 'Acc'))

model.save_weights(r'C:\Users\d19fd\Documents\tf.Keras\model\IMDB.weight')