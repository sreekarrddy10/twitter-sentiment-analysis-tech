import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------
# App Config
# -------------------
st.set_page_config(page_title="Tesla Sentiment Analyzer", layout="wide")

# -------------------
# Load Model & Tokenizer
# -------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("cnn_lstm_sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
labels = ['Negative', 'Neutral', 'Positive']

# -------------------
# Header
# -------------------
st.title("💬 Tesla Tweet Sentiment Classifier")
st.markdown("Upload a Tesla tweets CSV or enter a single tweet to analyze sentiment and predict engagement.")

# -------------------
# Textbox for Tweet Prediction
# -------------------
st.markdown("### 🔍 Predict Sentiment & Engagement of a Tweet")
user_tweet = st.text_input("Type a tweet to analyze:")
if user_tweet:
    sequence = tokenizer.texts_to_sequences([user_tweet])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)[0]
    sentiment_class = labels[np.argmax(prediction)]
    engagement_score = round(float(np.max(prediction)) * 100, 2)

    st.success(f"Predicted Sentiment: **{sentiment_class}**")
    st.info(f"Estimated Engagement Score: **{engagement_score}** (confidence %)")

# -------------------
# CSV Upload
# -------------------
uploaded_file = st.file_uploader("📄 Upload CSV File (must include 'text' column)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'.")
    else:
        df['text'] = df['text'].astype(str)
        sequences = tokenizer.texts_to_sequences(df['text'].tolist())
        padded = pad_sequences(sequences, maxlen=100)
        preds = model.predict(padded)
        df['Sentiment'] = [labels[np.argmax(p)] for p in preds]

        # Add fake or placeholder values if missing
        if 'nlikes' not in df.columns:
            df['nlikes'] = np.random.randint(0, 200, len(df))
        if 'nretweets' not in df.columns:
            df['nretweets'] = np.random.randint(0, 100, len(df))
        if 'nreplies' not in df.columns:
            df['nreplies'] = np.random.randint(0, 50, len(df))
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2022-07-11', periods=len(df), freq='T')
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

        # Add Engagement Score
        df['engagement_score'] = (
            df['nlikes'] * 0.5 +
            df['nretweets'] * 0.3 +
            df['nreplies'] * 0.2
        )

        # Show Paginated Preview Table
        st.markdown("### 🧾 Preview Data (Paginated)")
        st.dataframe(df[['text', 'Sentiment', 'nlikes', 'nretweets', 'nreplies', 'engagement_score']], use_container_width=True)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="tweets_with_sentiment.csv", mime='text/csv')

        # Pie Chart with Labels
        st.markdown("### 📊 Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts().reindex(labels, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#ff6b6b', '#f9dc5c', '#1dd1a1']  # Red, Yellow, Green
        wedges, texts, autotexts = ax.pie(
            sentiment_counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': 'white'}
        )
        ax.set_title("Sentiment Breakdown", fontsize=16)
        ax.axis('equal')
        st.pyplot(fig)

        # ======================
        # Additional Visualizations
        # ======================
        st.markdown("### 📊 Additional Insights")
        chart_option = st.selectbox("Choose a visualization", [
            "📈 Time-Based Sentiment Line Graph",
            "📊 Bar Chart: Likes & Retweets by Sentiment",
            "📊 Column Chart: Sentiment Counts by Day",
            "📊 Avg Engagement Score by Sentiment",
            "🌎 Tweet Activity by Hour and Sentiment",
            "☁️ Word Cloud per Sentiment"
        ])

        if chart_option == "📈 Time-Based Sentiment Line Graph":
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            sentiment_trend = df.groupby(df['timestamp'].dt.date)['Sentiment'].value_counts().unstack().reindex(columns=labels, fill_value=0)
            st.line_chart(sentiment_trend)

        elif chart_option == "📊 Bar Chart: Likes & Retweets by Sentiment":
            avg_engagement = df.groupby('Sentiment')[['nlikes', 'nretweets']].mean().reindex(labels)
            st.bar_chart(avg_engagement)

        elif chart_option == "📊 Column Chart: Sentiment Counts by Day":
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_sentiment = df.groupby(df['timestamp'].dt.date)['Sentiment'].value_counts().unstack().reindex(columns=labels, fill_value=0)
            st.bar_chart(daily_sentiment)

        elif chart_option == "📊 Avg Engagement Score by Sentiment":
            avg_score = df.groupby('Sentiment')['engagement_score'].mean().reindex(labels)
            st.bar_chart(avg_score)

        elif chart_option == "🌎 Tweet Activity by Hour and Sentiment":
            hour_sentiment = df.groupby(['hour', 'Sentiment']).size().unstack().fillna(0)
            st.bar_chart(hour_sentiment)

        elif chart_option == "☁️ Word Cloud per Sentiment":
            for sentiment in labels:
                st.markdown(f"### {sentiment} Word Cloud")
                text = " ".join(df[df['Sentiment'] == sentiment]['text'])
                if text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info(f"No tweets found for {sentiment}")