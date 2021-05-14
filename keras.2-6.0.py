import util2 as u

(x_train, x_test), (y_train, y_test) = u.mnist_data()
model = u.mnist_model()
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)

ret = model.train_on_batch(x_train[0:64], y_train[0:64])
print('\ntrain :', ret)

ret = model.test_on_batch(x_test[0:32], y_test[0:32])
print('\ntest : ', ret)

ret = model.predict_on_batch(x_test[-3:])
print('\npredict :\n', ret.round(1))