import util2 as u
from tensorflow.keras import backend as K

(x_train, x_test), (y_train, y_test) = u.mnist_data()
model = u.mnist_model()

def my_mse (y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

model.compile(optimizer='rmsprop',
              loss=my_mse,
              metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=128)
print('evaluation :', model.evaluate(x_test, y_test, verbose=0))