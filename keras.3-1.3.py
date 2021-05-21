from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import Constant
import tensorflow.keras

cnn = Sequential()

cnn.add(Conv2D(1,
               (3,3),
               kernel_initializer = tensorflow.keras.initializers.Ones(),
               bias_initializer = Constant(value=3),
               input_shape=(5,5,1)))

print(cnn.get_weights())