import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

cwd = os.getcwd()

# Charger le modèle
model = tf.keras.models.load_model(cwd + '/chatbot_model.h5')

# Charger le tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df)

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
