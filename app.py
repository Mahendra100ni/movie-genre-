import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

@st.cache_resource
def load_model():
    emb_model = joblib.load('emb_model.pkl')
    clf = joblib.load("logisticModelMovieGenereClassifier.pkl")
    return emb_model, clf

emb_model, clf = load_model()

def emb_dedo(sentence):
    return np.array(emb_model.encode(sentence))

st.title("🎬 Movie Genre Classifier")
user_input = st.text_area("Enter movie description:")

if st.button("Predict Genre"):
    if user_input:
        vec = emb_dedo(user_input).reshape(1, -1)
        probs = (clf.predict_proba(vec) * 100).astype(int)[0]
        classes = clf.classes_
        result = [classes[i] for i, p in enumerate(probs) if p > 5]

        st.success(f"Predicted Genre(s): {', '.join(result)}")
    else:
        st.warning("Please enter a description.")
