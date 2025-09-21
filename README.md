# Mini-Sentiment-Analysis-Web-App

📊 Sentiment Analyzer Web App

🔹 Project Overview

-This project is a Streamlit-based Sentiment Analysis Web Application that allows users to:
Upload a CSV file of text data and analyze sentiments (Positive, Negative, Neutral).
Enter a single text input and get an instant sentiment prediction.
View detailed reports and visualizations including sentiment distribution, word clouds, compound score histograms, and time trends.
Subscribe via email to receive updates (integrated with SendGrid API).
The app is built using Python, Streamlit, VADER Sentiment Analyzer, NLTK, Pandas, Matplotlib, and WordCloud.


⚙️ Features Implemented

✅ CSV Upload (columns: text, label, supports positive/negative or 0/1).
✅ Text Preprocessing (lowercase, remove punctuation).
✅ Vectorization using TfidfVectorizer.
✅ Model Training with LogisticRegression on 80/20 train/test split.
✅ Metrics Displayed: accuracy, precision, recall, F1, and class balance.
✅ Sample Predictions: a few true vs predicted examples from the test set.
✅ Single Review Prediction via /predict endpoint or frontend text box.


📂 Project Structure
mini-sentiment-app/
 ┣ 📄 app.py                # Flask backend with endpoints
 ┣ 📄 model_utils.py        # Preprocessing function
 ┣ 📄 requirements.txt      # Dependencies
 ┣ 📄 README.md             # Project documentation
 ┣ 📂 templates/
 ┃ ┗ 📄 index.html          # Frontend (HTML + minimal JS)
 ┣ 📂 static/
 ┃ ┗ 📄 styles.css          # CSS styles
 ┣ 📂 sample_data/
 ┃ ┗ 📄 sample_reviews.csv  # Example dataset
 ┣ 📂 tests/
 ┃ ┗ 📄 test_app.py         # Basic pytest suite
 ┗ 📂 screenshots/          # Screenshots or demo GIF


 1️⃣ Clone the Repository
git clone https://github.com/geethika-id25/mini-sentiment-app.git
cd mini-sentiment-app

2️⃣ Create Virtual Environment & Install Requirements
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Run the Flask App
python app.py


🗂️ Dataset Used

Sample CSV provided: sample_data/sample_reviews.csv

Columns:

text: review text

label: sentiment (positive/negative or 1/0)


📌 Approach

Pipeline: TfidfVectorizer → LogisticRegression

Preprocessing: lowercase, punctuation removal.

Evaluation: 80/20 split, metrics computed with scikit-learn.

Frontend: Minimal HTML/CSS/JS for usability.

Backend: Flask with /upload, /train, /predict endpoints.

