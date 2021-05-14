from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
#from keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.gcf().set_size_inches(15, 4)  #←設定圖形的寬和高 (英吋)
for i in range(5):
    ax = plt.subplot(1, 5, 1+i)  #←設定 1x5 的子圖表, 目前要畫第 1+i 個
    ax.imshow(train_images[i], cmap= 'binary')   #←顯示灰階圖片(黑底白字)
    ax.set_title('label = '+str(train_labels[i]), fontsize=18)  #←設定標題
plt.show()  #將圖形顯示出來