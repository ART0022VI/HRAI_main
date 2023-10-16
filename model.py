import tensorflow as tf
from tensorflow import keras

class QAModel(tf.keras.Model):

    def __init__(self):
        super(QAModel, self).__init__()
        self.embedding = keras.layers.Embedding(input_dim=1000, output_dim=64)
        self.lstm = keras.layers.LSTM(64)
        self.dense = keras.layers.Dense(1000, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x