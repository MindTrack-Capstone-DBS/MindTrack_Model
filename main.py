from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import logging

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

model_path = 'best_model_lstm.keras'
tokenizer_path = 'tokenizer.pickle'
label_encoder_path = 'label_encoder.pickle'
recommendation_path = 'mentalhealthtreatment.csv'

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    model = None

try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Tokenizer and label encoder loaded successfully.")
except Exception as e:
    print("Failed to load tokenizer/encoder:", e)
    tokenizer, label_encoder = None, None

try:
    df_rekomendasi = pd.read_csv(recommendation_path)
    print("Recommendation CSV loaded successfully.")
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

    prediction = model.predict(padded, verbose=0)  # Disable prediction logging
    label_index = np.argmax(prediction)
    label = label_encoder.inverse_transform([label_index])[0]
    recommendations = get_recommendations_by_status(label, df_rekomendasi)

    return {
        "prediction": label,
        "recommendations": recommendations if recommendations else ["No recommendations found."]
    }
