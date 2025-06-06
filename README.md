# twitter-sentiment-analysis-tech
# Twitter Sentiment Analysis & Engagement Dashboard

![GitHub Repo stars](https://img.shields.io/github/stars/sreekarrddy10/twitter-sentiment-analysis-tech)
![GitHub forks](https://img.shields.io/github/forks/sreekarrddy10/twitter-sentiment-analysis-tech)
![GitHub last commit](https://img.shields.io/github/last-commit/sreekarrddy10/twitter-sentiment-analysis-tech)
![License](https://img.shields.io/github/license/sreekarrddy10/twitter-sentiment-analysis-tech)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)

---

## 📄 Project Overview

This project presents an **AI-powered Twitter Sentiment & Engagement Dashboard**, designed to analyze sentiment trends, engagement patterns, and key insights from Twitter data. 

The solution leverages a **CNN+LSTM deep learning model** to classify tweet sentiments and predict engagement metrics, visualized through an interactive **Streamlit dashboard**.

---

## 🎯 Features

✅ End-to-end pipeline from raw tweets → analyzed insights  
✅ Sentiment Analysis (Positive / Negative / Neutral)  
✅ Engagement Prediction (Likes, Retweets, Replies)  
✅ Time-based trends visualization  
✅ Word clouds per sentiment class  
✅ Upload your own tweet datasets  
✅ Interactive dashboard for exploration  
✅ Powered by OpenAI + TensorFlow + Streamlit  

---

## 🏗️ Project Structure

```plaintext
├── dashboard/
│   ├── app.py                  # Main Streamlit app
│   ├── requirements.txt        # Project dependencies
│   ├── README.md                # Dashboard section docs
│   ├── scripts/                # Scripts related to dashboard
├── data/
│   ├── processed/              # Processed CSV files
│   ├── raw/                    # Raw tweet datasets
│   ├── notebooks/              # Exploratory analysis notebooks
├── models/
│   ├── cnn_lstm_sentiment_model.h5  # Trained LSTM model
│   ├── tokenizer.pkl           # Tokenizer used in training
├── visuals/                    # Wordclouds, bar charts, line plots
├── src/                        # Source code for preprocessing, training
├── streamlit_app/              # Optional custom assets
├── requirements.txt            # Project-wide requirements
└── README.md                   # Main project documentation
🚀 Project Workflow

1️⃣ Data Preparation
	•	Source: Public Twitter datasets
	•	Cleaned & preprocessed tweets
	•	Split into training & test sets

2️⃣ Model Training
	•	CNN + LSTM architecture using Keras / TensorFlow
	•	Word embeddings from tokenizer
	•	Trained on 25K+ labeled tweets
	•	Metrics: Accuracy ~ 92-94% on validation set

3️⃣ Engagement Scoring
	•	Formula used:
engagement_score = (likes * 0.5) + (retweets * 0.3) + (replies * 0.2)
	•	Dynamic scores visualized per sentiment class

4️⃣ Dashboard Deployment
	•	Built with Streamlit
	•	Real-time sentiment prediction
	•	Upload CSV → view live analysis
	•	Interactive charts & wordclouds

⸻

📊 Visualizations

✅ Time-series sentiment trends
✅ Likes & Retweets bar charts
✅ Engagement by sentiment type
✅ Word clouds for frequent terms
✅ Tweet-level analysis with predicted scores

⸻

⚙️ Tech Stack
	•	Python 3.10+
	•	TensorFlow / Keras
	•	NumPy / Pandas
	•	Matplotlib / Seaborn / Plotly
	•	Streamlit
	•	TextBlob / NLTK / SpaCy (preprocessing)
	•	Scikit-learn
	•	Wordcloud generation
	•	Git + GitHub (version control)
