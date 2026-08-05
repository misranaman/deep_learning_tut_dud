import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

model: Sequential = load_model("next_word_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer: Tokenizer = pickle.load(file)

reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}

max_length = 44


def generate_text(seed_text, num_words=10):
    text = seed_text
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([seq], maxlen=max_length, padding="pre")
        predictions = model.predict(padded, verbose=0)
        pos = np.argmax(predictions)
        next_words = reverse_index.get(pos, " ")
        text += " " + next_words
    return text


st.title("Next Word Prediction with Deep Learning")
seed = st.text_input("Enter a starting text:", "Hello")

num_of_words = st.slider("Number of words to generate", 1, 20, 10)

if st.button("Generate"):
    result = generate_text(seed, num_of_words)
    st.write(result)

