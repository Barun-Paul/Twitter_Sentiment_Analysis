import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import requests
import json
from datetime import datetime


@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')


@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"


def get_tweets_with_api(username, bearer_token, max_results=10):
    """
    Get tweets using Twitter API v2
    Note: This requires a Twitter API bearer token
    """
    try:
        
        user_url = f"https://api.twitter.com/2/users/by/username/{username}"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        
        user_response = requests.get(user_url, headers=headers)
        if user_response.status_code != 200:
            return None
            
        user_data = user_response.json()
        user_id = user_data['data']['id']
        
        # Get tweets
        tweets_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
        params = {
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics"
        }
        
        tweets_response = requests.get(tweets_url, headers=headers, params=params)
        if tweets_response.status_code == 200:
            return tweets_response.json()
        else:
            return None
            
    except Exception as e:
        st.error(f"Error fetching tweets: {str(e)}")
        return None


def create_card(tweet_text, sentiment, created_at=None, metrics=None):
    color = "green" if sentiment == "Positive" else "red"
    
    
    date_str = ""
    if created_at:
        try:
            date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            date_str = f"<small style='color: white;'>Posted: {date_obj.strftime('%Y-%m-%d %H:%M')}</small><br>"
        except:
            pass
    
    
    metrics_str = ""
    if metrics:
        metrics_str = f"""
        <small style='color: white;'>
            ❤️ {metrics.get('like_count', 0)} | 
            🔄 {metrics.get('retweet_count', 0)} | 
            💬 {metrics.get('reply_count', 0)}
        </small>
        """
    
    card_html = f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h5 style="color: white; margin: 0 0 10px 0;">{sentiment} Sentiment</h5>
        <p style="color: white; margin: 0 0 10px 0; line-height: 1.4;">{tweet_text}</p>
        {date_str}
        {metrics_str}
    </div>
    """
    return card_html


def get_sample_tweets():
    """Return sample tweets for demonstration when API is not available"""
    return [
        {
            "text": "I'm so excited about the new product launch! This is going to be amazing! 🚀",
            "created_at": datetime.now().isoformat(),
            "metrics": {"like_count": 45, "retweet_count": 12, "reply_count": 8}
        },
        {
            "text": "The customer service at this company is absolutely terrible. Never buying from them again.",
            "created_at": datetime.now().isoformat(),
            "metrics": {"like_count": 23, "retweet_count": 5, "reply_count": 15}
        },
        {
            "text": "Just finished reading an incredible book. Highly recommend it to everyone!",
            "created_at": datetime.now().isoformat(),
            "metrics": {"like_count": 67, "retweet_count": 18, "reply_count": 12}
        },
        {
            "text": "This movie was a complete waste of time and money. Don't bother watching it.",
            "created_at": datetime.now().isoformat(),
            "metrics": {"like_count": 34, "retweet_count": 7, "reply_count": 22}
        },
        {
            "text": "Amazing weather today! Perfect for a picnic in the park with friends.",
            "created_at": datetime.now().isoformat(),
            "metrics": {"like_count": 89, "retweet_count": 25, "reply_count": 14}
        }
    ]


def main():
    st.title("Twitter Sentiment Analysis")
    st.markdown("---")

    
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    
    option = st.selectbox("Choose an option", ["Input tweet", "Get tweets from user", "Demo with sample tweets"])
    
    if option == "Input tweet":
        st.subheader("Analyze Individual Tweet")
        text_input = st.text_area("Enter tweet to analyze sentiment", height=100)
        if st.button("Analyze Sentiment"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                card_html = create_card(text_input, sentiment)
                st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze.")

    elif option == "Get tweets from user":
        st.subheader("Analyze User's Tweets")
        
        
        st.info("⚠️ **Note**: Twitter API access requires authentication. If you don't have API credentials, use the 'Demo with sample tweets' option.")
        
        bearer_token = st.text_input("Enter Twitter API Bearer Token (optional)", type="password")
        username = st.text_input("Enter Twitter username (without @)")
        
        if st.button("Fetch and Analyze Tweets"):
            if not username.strip():
                st.warning("Please enter a username.")
            elif not bearer_token:
                st.error("Twitter API Bearer Token is required to fetch real tweets. Please use the 'Demo with sample tweets' option instead.")
            else:
                with st.spinner("Fetching tweets..."):
                    tweets_data = get_tweets_with_api(username, bearer_token)
                    
                    if tweets_data and 'data' in tweets_data:
                        st.success(f"Found {len(tweets_data['data'])} tweets!")
                        
                        for tweet in tweets_data['data']:
                            tweet_text = tweet['text']
                            sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                            
                            # Get metrics if available
                            metrics = tweet.get('public_metrics', {})
                            
                            card_html = create_card(
                                tweet_text, 
                                sentiment, 
                                tweet.get('created_at'),
                                metrics
                            )
                            st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        st.error("Could not fetch tweets. Please check your API credentials and username.")

    elif option == "Demo with sample tweets":
        st.subheader("Demo with Sample Tweets")
        st.info("This option uses sample tweets to demonstrate the sentiment analysis functionality.")
        
        if st.button("Load Sample Tweets"):
            sample_tweets = get_sample_tweets()
            
            for tweet in sample_tweets:
                sentiment = predict_sentiment(tweet['text'], model, vectorizer, stop_words)
                card_html = create_card(
                    tweet['text'], 
                    sentiment, 
                    tweet['created_at'],
                    tweet['metrics']
                )
                st.markdown(card_html, unsafe_allow_html=True)

    
    st.markdown("---")
    st.markdown("""
    ### How to get Twitter API access:
    1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
    2. Apply for a developer account
    3. Create a new app
    4. Generate a Bearer Token
    5. Use the Bearer Token in the app
    """)

if __name__ == "__main__":
    main()
