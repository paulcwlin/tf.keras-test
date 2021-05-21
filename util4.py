import matplotlib.pyplot as plt

# 繪製線圖 (可將訓練時所傳回的損失值或準確率等歷史記錄繪製成線圖)
# history: 內含一或多筆要繪資料的字典, 例如：{'loss': [4,2,1,…], 'acc': [2,3,5,…]}
# keys: 以 tuple 或串列指定 history 中要繪製的 key 值, 例如：('loss', 'acc')
# title: 以字串指定圖表的標題文字
# xyLabel: 以 tuple 或串列指定 x, y 軸的說明文字, 例如：('epoch', 'Accuracy')
# ylim: 以 tuple 或串列指定 y 軸的最小值及最大值, 例如 (1, 3), 超出範圍的值會被忽略
# size: 以 tuple 指定圖的尺寸, 預設為 (6, 4) (即寬 6 高 4 英吋)
def plot(history_dict, keys, title=None, xyLabel=[], ylim=(), size=()):
    lineType = ('-', '--', '.', ':')    # 線條的樣式, 畫多條線時會依序採用
    if len(ylim)==2: plt.ylim(*ylim)    # 設定 y 軸最小值及最大值
    if len(size)==2: plt.gcf().set_size_inches(*size)  # size預設為 (6,4)
    epochs = range(1, len(history_dict[keys[0]])+1)  # 計算有幾週期的資料
    for i in range(len(keys)):   # 走訪每一個 key (例如 'loss' 或 'acc' 等)
        plt.plot(epochs, history_dict[keys[i]], lineType[i])  # 畫出線條
    if title:   # 是否顯示標題欄
        plt.title(title)
    if len(xyLabel)==2:  # 是否顯示 x, y 軸的說明文字
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc='best') #upper left')  # 顯示圖例 (會以 key 為每條線的說明)
    plt.show()  # 顯示出畫好的圖

from tensorflow.keras.utils import to_categorical

# 將序列資料轉 one-hot 編碼的 generator 函式
# data: 序列訓練資料。2D陣列, 內容須為整數值, 例如 : [[0, 0, 4, 5],[0, 6, 7, 2]]
# y: 目標資料。2D 陣列, 例如 : [[0,1],[1,0]]
# batch_size: 用來指定生成器的批次量大小, 預設為 128
# num_classes: 序列的總類別數, 預設為 10000
# categorical: 是否要將目標資料依訓練資料的規則轉成 one-hot 格式, 預設為 False
def seq2oh_generator(data, y, batch_size=128, num_classes=10000, categorical=False):
    i=0
    while True:
        if i*batch_size >= len(data):
            i=0    #←若這一批次會超過資料長度, 則將 i 歸零
        start = i*batch_size    #←起始位置
        end = min(start+batch_size,len(data))    #←結束位置
        samples = to_categorical(data[start:end], num_classes=num_classes)
        if categorical:
            targets = to_categorical(y[start:end], num_classes=num_classes)   
        else:
            targets = y[start:end]
        i+=1
        
        yield samples, targets

# 將序列格式轉為 RNN 可接受的 shape
# seq:原始序列資料
def seq_for_rnn(seq):
    r_seq = seq.reshape(len(seq),-1,1)
    return r_seq

import numpy as np

# 將序列資料轉成 BOW 的 generator 
def bow_generator(data,y,batch_size=200,num_words=10000,bag_size=10):
    """
    將序列資料轉成BOW的 generator 
    """
    i = 0
    while True:
        if i*batch_size+batch_size > len(data)-1:
            i = 0   #←若這一批次會超過資料長度, 則將 i 歸零
        samples = np.zeros((batch_size,data.shape[1]//bag_size,num_words))   #←建立要輸出的 array, 初始的內容全為 0
        
        sequences = data[(i*batch_size):(i*batch_size+batch_size)]   #←先取出一個 batch 的資料
        for j,sequence in enumerate(sequences):
            for k in range(data.shape[1]//bag_size):
                word = sequence[k*bag_size:k*bag_size+bag_size]   #←取出一個 bag 的詞
                samples[j,k,word] = 1.   #←依據出現的詞將輸出 array 的對應位置設定為1
                
        targets = y[(i*batch_size):(i*batch_size+batch_size)]
        i+=1

        yield samples, targets

def n_gram(data,n=1,num_words=10000,index=None,append=True):
    if index ==None:
        index = {}   #←若詞彙對照表為 None, 建立一個空的對照表 
    samples = []   #←輸出資料, 一開始為空 list
    for seq in data:
        new_seq = seq.copy()   #←複製一個序列
        for n_size in range(2,n+1):
            for i in range(len(seq)-(n_size-1)):
                word = tuple(seq[i:i+n_size])   #←找出 N 以下的所有連續組合
                if word in index:
                    new_seq.append(index[word])  #←如果這個組合已經在詞彙對照表中, 則將對應的號碼加入資料中
                elif append:
                    num_words+=1
                    index[word] = num_words
                    new_seq.append(num_words)  #←如果詞彙對照表中無此組合, 則為這個組合建立編號並加入對照表中, 再將此編號加進資料中
                else:
                    new_seq.append(0)  #←如果詞彙對照表中無此組合, 則在資料中加入 0
                
        samples.append(new_seq)  #←將擴增完的資料加進輸出資料中

    samples = np.array(samples)
    return samples,index,num_words 


