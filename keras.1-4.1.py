from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model  #←匯入 load_model 函式

#載入 MNIST 資料集
(_, _), (test_images, test_labels) = mnist.load_data()

#資料預處理
x_test = test_images.reshape((10000, 28 * 28)) #←將 10000 筆測試樣本做同樣的轉換
x_test = x_test.astype('float32') / 255       #←將 10000 筆測試標籤做同樣的轉換
y_test  = to_categorical(test_labels)

model = load_model('MnistModel.h5')  #← 由檔案載入模型

test_loss, test_acc = model.evaluate(x_test, y_test) #←用測試資料評估成效
print('對測試資料的準確率：', test_acc)