import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

df = pd.read_csv('dataset_review_tokped_labelled.csv')

print(df)
max_features = 2000
max_len = 200

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['Review'].values)

sentiment_labels = ['negative','neutral','positive']

filename = 'finalized_model_modif.sav'
loaded_model = pickle.load(open(filename, 'rb'))


def prediksiModel(text):
    text_seq = tokenizer.texts_to_sequences([text])
    print(text_seq)
    text_seq = pad_sequences(text_seq, maxlen=200)
    pred = loaded_model.predict(text_seq)[0]
    print(text_seq)
    predict = sentiment_labels[np.argmax(pred)]
    print('Text:', text)
    print(predict)
    st.subheader("Hasil Sentimen: "+ predict)
    

st.title('Sentimen Analisis Menggunakan LSTM Terhadap Review Produk Makanan di Tokopedia')


teks = st.text_input("Masukkan Teks: ")
clicked = st.button("Prediksi Teks")

if clicked:
    prediksiModel(teks)

