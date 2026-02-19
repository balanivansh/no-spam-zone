import streamlit as st
import pickle
import string
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Ensure required NLTK data is available on startup (helps when deploying to clean hosts)
# quiet=True avoids verbose output during app startup
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Cache stopwords once (may raise LookupError if downloads failed)
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = None


def transform_text(text):
    if not text:
        return ""

    text = text.lower()
    
    # Try NLTK tokenization, fall back to simple split if NLTK data not available
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # Fallback: simple whitespace and punctuation split
        import re
        tokens = re.findall(r'\b\w+\b', text)

    # keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]

    # remove stopwords and punctuation
    if STOPWORDS is not None:
        tokens = [t for t in tokens if t not in STOPWORDS]
    else:
        tokens = [t for t in tokens if t not in string.punctuation]

    # stem tokens
    stemmed = [ps.stem(t) for t in tokens]
    return " ".join(stemmed)


# Safe loading of model artifacts: show friendly message in Streamlit if missing
VECT_PATH = Path('vectorizer.pkl')
MODEL_PATH = Path('model.pkl')

tfidf = None
model = None
if VECT_PATH.exists() and MODEL_PATH.exists():
    try:
        tfidf = pickle.load(open(VECT_PATH, 'rb'))
    except Exception as e:
        tfidf = None
        st.warning(f"Could not load vectorizer.pkl: {e}")
    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
    except Exception as e:
        model = None
        st.warning(f"Could not load model.pkl: {e}")
else:
    # Defer messaging until UI renders (so Streamlit can display it)
    pass


st.title("Email/SMS Spam Classifier")

if not VECT_PATH.exists() or not MODEL_PATH.exists():
    st.error("Model artifacts missing: please add 'vectorizer.pkl' and 'model.pkl' to the project root.")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if tfidf is None or model is None:
        st.error("Model is not loaded. Ensure 'vectorizer.pkl' and 'model.pkl' exist and are valid pickles.")
    elif not input_sms:
        st.info("Please enter a message to classify.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
