import json
import numpy as np
import keras as keras
from keras.models import *
from keras.layers import Dense
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
with open('dataset.json', encoding="utf-8") as file:
    data = json.load(file)
    trainQ = []
    trainA = []
    for i in range(len(data["Q"])):
        trainQ.append(data["Q"][i])
        trainA.append(data["A"][i])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainQ + trainA)
prepQ = tokenizer.texts_to_sequences(trainQ)
prepA = tokenizer.texts_to_sequences(trainA)
maxlen = max([len(x) for x in trainQ + trainA])
readyQ = pad_sequences(prepQ, maxlen=maxlen)
readyA = pad_sequences(prepA, maxlen=maxlen)

model = Sequential([
    layers.Input(shape=maxlen, name="input"),
    layers.Embedding(input_dim=100, output_dim=10),
    LSTM(68, activation='tanh', return_sequences=True, name="main"),
    layers.LSTM(68, activation='softmax', name="output")
])

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(readyQ, readyA, epochs=2000, batch_size=32)
model.save("test.model")
