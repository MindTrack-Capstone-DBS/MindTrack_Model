from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import string
import nltk
import os
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from keras.layers import Input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Initialize TensorFlow session
tf.keras.backend.clear_session()

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

model_path = 'best_model_lstm.keras'
tokenizer_path = 'tokenizer.pickle'
label_encoder_path = 'label_encoder.pickle'
recommendation_path = 'mentalhealthtreatment.csv'

# Initialize global variables
model = None
tokenizer = None
label_encoder = None
df_rekomendasi = None

def load_resources():
    global model, tokenizer, label_encoder, df_rekomendasi
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info("Tokenizer and label encoder loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer/encoder: {e}")
        tokenizer, label_encoder = None, None

    try:
        df_rekomendasi = pd.read_csv(recommendation_path)
        logger.info("Recommendation CSV loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load recommendation CSV: {e}")
        df_rekomendasi = None

# Load resources at startup
load_resources()

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
    if recommendation_df is None:
        return ["No recommendations available"]
    filtered_df = recommendation_df[recommendation_df['status'] == status]
    return filtered_df['treatment'].tolist() if not filtered_df.empty else ["No specific recommendations found"]

@app.post("/predict")
async def predict(input: TextInput):
    if model is None or tokenizer is None or label_encoder is None:
        return {"error": "Model or resources not loaded properly"}

    try:
        tokens = preprocess(input.text)
        sequence = tokenizer.texts_to_sequences([" ".join(tokens)])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=100,
            padding='post',
            truncating='post'
        )

        with tf.device('/CPU:0'):
            prediction = model.predict(padded, verbose=0)
        
        label_index = np.argmax(prediction)
        label = label_encoder.inverse_transform([label_index])[0]
        recommendations = get_recommendations_by_status(label, df_rekomendasi)

        return {
            "prediction": label,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": "An error occurred during prediction"}
