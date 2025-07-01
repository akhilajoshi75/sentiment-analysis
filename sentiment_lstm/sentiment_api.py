# sentiment_lstm/sentiment_api.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import joblib

from config import MODEL_PATH, TOKENIZER_PATH

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model and tokenizer
model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

tokenizer = joblib.load(TOKENIZER_PATH)
max_len = 200


@app.route("/", methods=["GET"])
def home():
    return "Sentiment API is running!"

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text'}), 400

    text = data['text']
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad, verbose=0)[0][0]

    sentiment = 'positive' if pred > 0.5 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
