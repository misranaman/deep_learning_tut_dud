import streamlit as st
from streamlit.type_util import SupportsStr
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

## loading the tensorflow model for making prediction

model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer: Tokenizer = pickle.load(file)

st.title("Twitter Tweet Sentiment Analysis")

tweet: str | None = st.text_area("Enter the Tweet:")

if tweet is not None:
    if st.button("Predict Sentiment:") and tweet.strip():
        sequences = tokenizer.texts_to_sequences([tweet])
        sequences = pad_sequences(sequences, padding="post", maxlen=99)
        prediction = model.predict(sequences)
        predicted_class = np.argmax(prediction, axis=1)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        st.write("Sentiment", sentiment_map[predicted_class])
