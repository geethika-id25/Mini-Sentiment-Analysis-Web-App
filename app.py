import os
import joblib
from io import TextIOWrapper
from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model_utils import preprocess_text

MODEL_PATH = "sentiment_model.joblib"
VECT_PATH = "vectorizer.joblib"

app = Flask(__name__)

state = {"df": None, "model": None, "vectorizer": None, "metrics": None, "sample_test": None}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(TextIOWrapper(file.stream, encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    if not {"text", "label"}.issubset(df.columns):
        return jsonify({"error": "CSV must have 'text' and 'label' columns"}), 400

    def normalize_label(x):
        s = str(x).strip().lower()
        if s in ["1", "positive", "pos", "true", "yes"]:
            return "positive"
        return "negative"

    df["label"] = df["label"].apply(normalize_label)
    df["text"] = df["text"].astype(str).apply(preprocess_text)

    state["df"] = df
    counts = df["label"].value_counts().to_dict()
    return jsonify({"message": "Upload successful", "rows": len(df), "class_counts": counts})

@app.route("/train", methods=["POST"])
def train():
    df = state.get("df")
    if df is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_train_t, X_test_t = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label="positive", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label="positive", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label="positive", zero_division=0)),
    }

    samples = [{"text": t, "true": tr, "pred": pr} for t, tr, pr in zip(X_test[:5], y_test[:5], y_pred[:5])]

    state.update({"model": model, "vectorizer": vectorizer, "metrics": metrics, "sample_test": samples})
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)

    return jsonify({"message": "Training complete", "metrics": metrics, "sample_test": samples})

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text") if request.is_json else request.form.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prepped = preprocess_text(text)

    model, vectorizer = state.get("model"), state.get("vectorizer")
    if model is None or vectorizer is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
            model, vectorizer = joblib.load(MODEL_PATH), joblib.load(VECT_PATH)
            state["model"], state["vectorizer"] = model, vectorizer
        else:
            return jsonify({"error": "Model not trained"}), 400

    vec = vectorizer.transform([prepped])
    pred, proba = model.predict(vec)[0], float(model.predict_proba(vec).max())
    return jsonify({"text": text, "prediction": pred, "confidence": proba})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
