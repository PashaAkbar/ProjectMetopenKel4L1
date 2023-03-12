import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

# Load Dataset kedalam program
df = pd.read_csv('dataset_review_tokped_labelled.csv')

print(df)

# Inisiliasi max feature dan max_length
max_features = 2000
max_len = 200

# Tokenizer
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['Review'].values)
X = tokenizer.texts_to_sequences(df['Review'].values)
X = pad_sequences(X, maxlen=max_len)

# Split atau bagi data menjadi data train dan data test dengan rasio data train 80% dan data test 20%
Y = pd.get_dummies(df['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



# Buat Model LSTM
# model = Sequential()
# model.add(Embedding(max_features, 128, input_length=X.shape[1]))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(3, activation='softmax'))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=32, input_length=max_len),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(3, activation='sigmoid')
]
)


# model = Sequential()
# model.add(Embedding(max_features, 4))
# model.add(LSTM(4))
# model.add(Dense(3, activation='sigmoid'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Y_train = tf.one_hot(Y_train, 3)
# latih model
batch_size = 32
history = model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, validation_data=(X_test, Y_test))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_modif.png')
plt.show()



# Evaluasi Model
score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=2)
print('Test loss:', score)
print('Test accuracy:', acc)

filename = 'finalized_model_modif.sav'
pickle.dump(model, open(filename, 'wb'))
