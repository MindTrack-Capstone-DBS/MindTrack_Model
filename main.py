from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware



nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



model_path = 'best_model_lstm.keras'
tokenizer_path = 'tokenizer.pickle'
label_encoder_path = 'label_encoder.pickle'
recommendation_path = 'mentalhealthtreatment.csv'


try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")
except Exception as e:
    print("Failed to load model:", e)
    model = None

try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Tokenizer and label encoder loaded.")
except Exception as e:
    print("Failed to load tokenizer/encoder:", e)
    tokenizer, label_encoder = None, None

try:
    df_rekomendasi = pd.read_csv(recommendation_path)
    print("Rekomendasi CSV loaded.")
except Exception as e:
    print("Failed to load recommendation CSV:", e)
    df_rekomendasi = None


app = FastAPI()


class TextInput(BaseModel):
    text: str


def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

def get_recommendations_by_status(status, recommendation_df):
    filtered_df = recommendation_df[recommendation_df['status'] == status]
    return filtered_df['treatment'].tolist()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Atau ganti dengan domain express jika ingin ketat
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(input: TextInput):
    if model is None or tokenizer is None or label_encoder is None:
        return {"error": "Model or resources not loaded"}

    tokens = preprocess(input.text)

    sequence = tokenizer.texts_to_sequences([" ".join(tokens)])

    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=100,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(padded)
    label_index = np.argmax(prediction)
    label = label_encoder.inverse_transform([label_index])[0]
    recommendations = get_recommendations_by_status(label, df_rekomendasi)

    return {
        "prediction": label,
        "recommendations": recommendations if recommendations else ["Tidak ada rekomendasi ditemukan."]
    }
