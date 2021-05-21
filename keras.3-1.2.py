from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
import tensorflow.keras

cnn = Sequential()

cnn.add(Conv2D(1,
               (3,3),
               kernel_initializer = tensorflow.keras.initializers.Ones(),
               input_shape=(5,5,1)))

print(cnn.get_weights())
        
