import json
import numpy as np  
from model import QAModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras

print("================================")
print("Открываю файл...")
with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

questions = []
answers = []
print("Загружаю вопросы...")  
for q, a in zip(dataset['Q'], dataset['A']):
  questions.append(q)
  answers.append(a)
  
print("Токенизирую...")
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
q_seqs = tokenizer.texts_to_sequences(questions)
a_seqs = tokenizer.texts_to_sequences(answers)
print("Сиквенцирую...")
max_len = max(max(len(seq) for seq in q_seqs), max(len(seq) for seq in a_seqs))
q_seqs = pad_sequences(q_seqs, padding='post', maxlen=max_len)
a_seqs = pad_sequences(a_seqs, padding='post', maxlen=max_len)
print("Конвертирую в numpy массивы...")
q_seqs = np.array(q_seqs)
a_seqs = np.array(a_seqs)
print("Генерирую модель...")
model = QAModel()
print("Компилирую модель...")
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

print("Обучаю...")  
model.fit(q_seqs, a_seqs, epochs=10)

print("Сохраняю...")
model.save('qa_model')

print("Успех!")
print("================================")