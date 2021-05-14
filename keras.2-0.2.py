import util2 as u

(x_train, x_test), (y_train, y_test) = u.mnist_data()

model = u.mnist_model()

model.fit(x_train, y_train, epoch=5, batch_size=128)

