import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd

cwd = os.getcwd()


class CustomEmbeddingLayer(Layer):
    def __init__(self, input_dim, output_dim, input_length=None, **kwargs):
        super(CustomEmbeddingLayer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim, output_dim, input_length=input_length, **kwargs)

    def call(self, inputs):
        return self.embedding(inputs)


class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.embedding_layer = CustomEmbeddingLayer(input_dim, output_dim, input_length=max_seq_length)
        # Add other layers of your model here
        self.dense_layer = tf.keras.layers.Dense(...)

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        # Add your model logic here
        x = self.dense_layer(x)
        return x
    
    def build(self, input_shape):
        super(CustomModel, self).build(input_shape)
        self.dense_layer.build(input_shape=(None, output_dim))


df = pd.read_csv(cwd + '/dialogs.txt', sep='\t')

# Charger le tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df)

# Définir la longueur maximale de séquence
max_seq_length = 50

# Définir les dimensions de l'embedding
input_dim = len(tokenizer.word_index) + 1
output_dim = 100

# Créer une instance du modèle personnalisé
model = CustomModel()
model.build((None, output_dim))

# Charger les poids du modèle pré-entraîné
model.load_weights(cwd + '/chatbot_model.h5')

# Fonction pour prédire la réponse
def predict_answer(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_seq_length, padding='post')
    predicted_index = np.argmax(model.predict(input_sequence_padded), axis=-1)[0]
    predicted_word = tokenizer.index_word.get(predicted_index, '')
    return predicted_word

# Interface utilisateur avec Streamlit
st.title('Chatbot')
user_input = st.text_input('Vous:')

if st.button('Envoyer'):
    if user_input:
        bot_response = predict_answer(user_input)
        st.text_area('Chatbot:', value=bot_response, height=100)
