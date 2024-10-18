import streamlit as st
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = load_model('movie_review_model.h5')

max_len = 1451
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    if isinstance(text, list):

        text = " ".join(text)

    text = text.lower()

    text = re.sub(r"[^a-z!?',.\s ]", "", text)

    text = re.sub(r'\b(`|``)\b', '"', text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\b(im)\b","i'm",text)

    text = re.sub(r'\b(br|b)\b', '', text)

    text = re.sub(r"\b(\s's)/b","'s",text)

    tokens = word_tokenize(text)

    tokens = [token for token in tokens if token not in stop_words]

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    cleaned_text = " ".join(tokens)

    return cleaned_text

# Prediction function
def predict(new_text):
    cleaned_new_text = clean_text(new_text)
    new_sequence = tokenizer.texts_to_sequences([cleaned_new_text])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len,padding='post')
    prediction = model.predict(new_padded_sequence)
    return prediction[0][0]


# Title of the app
st.title("Movie Review Sentiment Classifier")

# Input box for user review
review = st.text_area("Enter a movie review:")

# Predict button
if st.button("Classify Review"):
    if review:
        prediction = predict(review)
        if prediction > 0.5:
            st.write("This is a **positive** review!")
        else:
            st.write("This is a **negative** review!")
    else:
        st.write("Please enter a review for classification.")
