# 📊 Twitter Sentiment Analysis

A Streamlit web app that performs sentiment analysis on tweets using a machine learning model. This project allows users to either input custom tweets or fetch live tweets from a public Twitter account using the `ntscraper` (Nitter) module, and displays whether the sentiment is **positive** or **negative**.

## 🚀 Features

- 🔍 Sentiment prediction on user-inputted tweets or scraped tweets from Twitter usernames.
- 🐦 Live tweet scraping using the `ntscraper` package (no Twitter API key needed).
- 🧠 Machine Learning model trained on tweet data using TF-IDF vectorization.
- 🧹 Text preprocessing: cleaning, lowercasing, tokenization, and stopword removal.
- 🎨 Stylish and interactive Streamlit UI with colored sentiment cards.
- ⚡ Fast loading and optimized resource usage with Streamlit caching.

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- scikit-learn
- nltk
- ntscraper

## 📂 Project Structure

```
.
├── app.py                          # Streamlit application for sentiment analysis
├── Twitter_sentiment_Analysis.ipynb  # Jupyter notebook for training and analysis
├── model.pkl                       # Trained ML model (e.g., SVM or Logistic Regression)
├── vectorizer.pkl                  # Trained TF-IDF vectorizer
├── requirements.txt                # Required Python packages
└── README.md                       # Project documentation
```

## 💡 How It Works

1. User selects between:
   - Input custom tweet
   - Fetch tweets via Twitter username
   - Use sample tweets (no API needed)
2. Tweets are preprocessed (cleaning, stopword removal)
3. Vectorized using pre-trained TF-IDF
4. Passed into a trained model to get prediction
5. Output rendered as a colored card with additional info

## 🔐 Using Twitter API (Optional)

To fetch live tweets:
1. Visit [Twitter Developer Portal](https://developer.twitter.com/)
2. Apply for a developer account and create an App
3. Generate a **Bearer Token**
4. Use this token and enter the Twitter username in the app

## 📥 Dataset Used

https://www.kaggle.com/datasets/kazanova/sentiment140

## 🖼️ Screenshot

![Output_2](https://github.com/user-attachments/assets/351fa7c6-6975-4e61-b09f-635cd6799751)
![Output_1](https://github.com/user-attachments/assets/f5c20207-d8fb-4df1-a28f-b00acdfa30f4)



## 📦 requirements.txt

```
streamlit
scikit-learn
nltk
ntscraper
```

## 👨‍💻 Author

**Barun Paul** — Developer and ML Enthusiast
