import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json

# Load cleaned tweets with sentiment
df = pd.read_csv("Tesla_with_sentiment_fresh.csv")

# Ensure only unlabeled tweets are processed
texts = df["cleaned_tweet"].astype(str).tolist()

# Load tokenizer
with open("tokenizer.json", "r") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

# Load trained model
model = load_model("cnn_lstm_sentiment_model.h5")

# Predict
predictions = model.predict(X, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Map labels to sentiment
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df["predicted_sentiment"] = [label_map[label] for label in predicted_labels]

# Save the result
df.to_csv("Tesla_sentiment_predicted.csv", index=False)
print("✅ Predictions saved to Tesla_sentiment_predicted.csv")