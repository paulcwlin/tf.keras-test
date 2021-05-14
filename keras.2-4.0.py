import util2 as u

(x_train, x_test), (y_train, y_test) = u.mnist_data()
model = u.mnist_model()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc', 'mse'])

model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)

t_loss, t_acc, t_mse = model.evaluate(x_test, y_test)

print('Test Data loss', t_loss)
print('Test Data Accuracy', t_acc)
print('Test Data MSE', t_mse)