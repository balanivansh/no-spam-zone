# No-Spam-Zone

SMS/Email spam classifier using NLP and a scikit-learn model. This small project provides a Streamlit web app (`app.py`) that loads a pre-trained TF-IDF vectorizer (`vectorizer.pkl`) and model (`model.pkl`) to classify messages as spam or not spam.

## Files

- `app.py` - Streamlit app that accepts text input and predicts spam/ham.
- `sms_spam_NLP.ipynb` - Notebook used for exploratory data analysis and model training.
- `spam2.csv` / `spam2_utf8.csv` - Dataset CSV files.
- `vectorizer.pkl`, `model.pkl` - Pretrained artifacts required by `app.py` (not present in repo snapshot). Add these to run the app.
- `requirements.txt` - Python dependencies.

## Setup

1. Create a Python environment (recommended: Python 3.8+).

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. NLTK data: the app uses `word_tokenize` and stopwords. Run once to download the required data:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. Ensure `vectorizer.pkl` and `model.pkl` are present in the project root (these are binary model artifacts produced by training).

## Run the app

Start Streamlit from the project folder:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`).

## Notes and troubleshooting

- The notebook contains shell-style Jupyter cells like `!pip install ...` which may trigger notebook lint warnings; prefer `%pip install` in modern notebooks.
- If you get errors about missing `vectorizer.pkl` or `model.pkl`, train the model using the notebook or copy the artifacts into the project.

## License

This project has no license specified. Add a `LICENSE` file if you want to set one.
