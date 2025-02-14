import os
import nltk
from flask import Flask, render_template, request
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import re

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'movie_review_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'model', 'tokenizer.pickle')

# Load tokenizer
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    raise Exception("Tokenizer file not found. Ensure 'tokenizer.pickle' is in the model directory.")

# Load model
try:
    model = load_model(MODEL_PATH)
except:
    raise Exception("Model file not found. Ensure 'movie_review_model.h5' is in the model directory.")

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
    text = re.sub(r"\b(im)\b", "i'm", text)
    text = re.sub(r'\b(br|b)\b', '', text)
    text = re.sub(r"\b(\s's)/b", "'s", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = " ".join(tokens)
    return cleaned_text

def predict(new_text):
    cleaned_new_text = clean_text(new_text)
    new_sequence = tokenizer.texts_to_sequences([cleaned_new_text])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=max_len, padding='post')
    prediction = model.predict(new_padded_sequence)
    return prediction[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        review = request.form['review']
        if review:
            prediction = predict(review)
            result = "positive" if prediction > 0.5 else "negative"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)