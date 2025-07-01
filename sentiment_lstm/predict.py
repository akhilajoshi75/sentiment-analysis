import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings


import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import joblib
from config import MODEL_PATH, TOKENIZER_PATH

max_len = 200

# Load tokenizer once (outside function)
tokenizer = joblib.load(TOKENIZER_PATH)

# Load and compile model
model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

def analyze_sentiment_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad, verbose=0)[0][0]

    # Simple binary classification
    return 'positive' if pred > 0.5 else 'negative'