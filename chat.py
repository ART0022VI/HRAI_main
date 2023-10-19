import json
import numpy as np
import keras as keras
from keras.models import *
from keras.layers import Dense
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
with open('dataset.json', encoding="utf-8") as file:
    data = json.load(file)
    trainQ = []
    trainA = []
    for i in range(len(data["Q"])):
        trainQ.append(data["Q"][i])
        trainA.append(data["A"][i])
model = load_model("test.model")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainQ + trainA)
maxlen = max([len(x) for x in trainQ + trainA])
while True:
    user_input = input("Введите ваш вопрос: ")
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=maxlen)
    print(user_input_padded)
    predicted_response = model.predict(user_input_padded)
    print(predicted_response.tolist())
    predicted_response_text = list(map(tokenizer.sequences_to_texts, predicted_response.tolist()))
    print("Ответ бота:", predicted_response_text)

