from model import QAModel
import json

import numpy as np
from tensorflow import keras

model = keras.models.load_model('qa_model')

with open('dataset.json') as f:
  dataset = json.load(f) 

questions = dataset['Q']
answers = dataset['A']

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)

while True:
  q = input("Вопрос: ")
  q_seq = tokenizer.texts_to_sequences([q])[0]
  a_idx = np.argmax(model.predict(q_seq[np.newaxis, :]))
  a = tokenizer.sequences_to_texts([[a_idx]])[0]
  print("Ответ: {}".format(a))