from fastapi import FastAPI
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
import os

# Set path lokal nltk_data
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# File paths
model_path = 'best_model_lstm.keras'
tokenizer_path = 'tokenizer.pickle'
label_encoder_path = 'label_encoder.pickle'
recommendation_path = 'mentalhealthtreatment.csv'

# Load model
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# Load tokenizer dan encoder
try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Tokenizer and label encoder loaded.")
except Exception as e:
    print("❌ Failed to load tokenizer/encoder:", e)
    tokenizer, label_encoder = None, None

# Load CSV rekomendasi
try:
    df_rekomendasi = pd.read_csv(recommendation_path)
    print("✅ Recommendation CSV loaded.")
except Exception as e:
    print("❌ Failed to load recommendation CSV:", e)
    df_rekomendasi = None

# FastAPI app
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti jika ingin spesifik domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class TextInput(BaseModel):
    text: str

# Preprocessing
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

# Rekomendasi
def get_recommendations_by_status(status, recommendation_df):
    if recommendation_df is None:
        return ["Rekomendasi tidak tersedia."]
    filtered_df = recommendation_df[recommendation_df['status'] == status]
    return filtered_df['treatment'].tolist()

# Route utama
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

# Jalankan untuk platform seperti Railway
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
