from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

df = pd.read_csv('CLEANED_DATA.csv')
x = df['Reviews']
y = df['Assumed_Sentiment']


X = []

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emotions
                           u"\U0001F300-\U0001F5FF"  # sumbols and pictographs
                           u"\U0001F680-\U0001F6FF"  # transport and map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags
                           u"\U00002700-\U000027BF"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
for i in x:
    X.append(emoji_pattern.sub('', i))

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.20, shuffle=True)


vocab_size = 40000
embedding_dim = 16
max_length = 250
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(x_train)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
testing_sentences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# model.summary()

training_labels_final = np.array(y_train)
testing_labels_final = np.array(y_test)

num_epochs = 20
history = model.fit(padded, training_labels_final, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels_final))

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))
# plt.plot(epochs, acc, 'r', 'Training Accuracy')
# plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
# plt.title('Training and validation accuracy')
# plt.figure()
# plt.plot(epochs, loss, 'r', 'Training Loss')
# plt.plot(epochs, val_loss, 'b', 'Validation Loss')
# plt.title('Training and validation loss')
# plt.figure()
# plt.show()

text = "Christmas at the park. We had a nice time even though the park was super crowded. Shops and Chocolate world had cute items to buy and the pretzels and cheese we snacked on were excellent. Parking and tram was a breeze. My only complaint is they only had one person working some stands (kettle corn literally had one person doing everything, we ended up leaving the line as it wasnâ€™t moving at all) and only one monorail was running. The operators werenâ€™t filling the cars either so we waited forever. Other than that the decor was lovely and all in all in was worth the trip."
text_seq = tokenizer.texts_to_sequences(text)
text_pad = pad_sequences(text_seq, maxlen=max_length)
ps = model.predict(text_pad).flatten()
print(ps)
