from tensorflow.keras.preprocessing.text import Tokenizer

samples = ['吃 什麼', '咖哩飯 還是 牛排 還是 麵包 還是 都 吃']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(samples)

print(tokenizer.word_counts)
print(tokenizer.index_word)
print(tokenizer.word_index)

seq = tokenizer.texts_to_sequences(['吃 咖哩飯', '牛排 和 麵包'])
print(seq)

text = tokenizer.sequences_to_texts(seq)
print(text)