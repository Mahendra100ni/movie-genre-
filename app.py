import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import joblib

@st.cache_resource
def load_model():
    emb_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # lightweight & fast
    clf = joblib.load("logisticModelMovieGenereClassifier.pkl")
    return emb_model, clf

emb_model, clf = load_model()

def emb_dedo(sentence):
    return np.array(emb_model.encode(sentence))

st.title("🎬 Movie Genre Classifier")

# Create text area tied to session_state["user_input"]
st.text_area("Enter movie description:", key="k")

# Access the value after user input
user_input = st.session_state["k"]

if st.button("Predict Genre"):
    if user_input:
        vec = emb_dedo(user_input).reshape(1, -1)
        probs = (clf.predict_proba(vec) * 100).astype(int)[0]
        classes = clf.classes_
        result = [classes[i] for i, p in enumerate(probs) if p > 5]
        st.success(f"Predicted Genre(s): {', '.join(result)}")
    else:
        st.warning("Please enter a description.")
