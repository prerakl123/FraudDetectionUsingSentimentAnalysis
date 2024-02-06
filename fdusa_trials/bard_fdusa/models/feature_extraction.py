import json
import re
import sqlite3

import dlib
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def download_transcript(video_id, api_key):
    # Implement functionality to download transcript from OpenAI Whisper using API key
    pass


def preprocess_text(text):
    """Preprocesses text for feature extraction."""
    text = text.lower()                                        # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation and special characters
    text = re.sub(r"\s+", " ", text)            # Remove extra whitespace
    return text


def extract_text_features(text):
    # TODO: get list of keywords related to fraud attempts
    with open("fraud_keywords.txt", "r") as f:
        fraud_keywords = f.read().split('\n')
        f.close()

    # Preprocess text
    processed_text = preprocess_text(text)

    # Calculate sentiment scores
    sentiment = SentimentIntensityAnalyzer().polarity_scores(processed_text)

    stop_words = stopwords.words("english")
    stop_word_count = len([word for word in processed_text.split() if word in stop_words])
    fraud_keyword_count = len([word for word in processed_text.split() if word in fraud_keywords])

    features = {
        "sentiment": sentiment,
        "stop_word_count": stop_word_count,
        "fraud_keyword_count": fraud_keyword_count
    }

    return features


def store_features(video_id, features):
    """Stores extracted features in a SQLite database."""
    conn = sqlite3.connect("fraud_data.db")
    c = conn.cursor()

    c.execute(
        "CREATE TABLE IF NOT EXISTS features ("
        "video_id TEXT, "
        "sentiment TEXT, "
        "stop_word_count INTEGER, "
        "fraud_keyword_count INTEGER"
        ")"
    )
    c.execute(
        "INSERT INTO features VALUES (?, ?, ?, ?)",
        (
            video_id,
            json.dumps(features["sentiment"]),
            features["stop_word_count"],
            features["fraud_keyword_count"]
        )
    )

    conn.commit()
    conn.close()


def extract_landmark_features(face_img):
    # Load dlib face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file!

    # Detect face and predict landmarks
    rects = detector(face_img, 1)
    for rect in rects:
        shape = predictor(face_img, rect)

        # Calculate distances and angles between specific landmarks (e.g., eyes, mouth)
        # based on your chosen features
        # ...

        # Return extracted features
        return features


if __name__ == '__main__':
    # https://bard.google.com/chat/278af56b5dfd2930
    pass
