import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download('stopwords')

# Load the NLP model
def load_model():
    dataset = pd.read_csv("C://Users//Dell//OneDrive//Desktop//xyz//Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
    corpus = []
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)

    return classifier, cv

# Function to predict sentiment
def predict_sentiment(review, model, cv):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    input_data = cv.transform([review]).toarray()
    prediction = model.predict(input_data)
    return prediction[0]

# Load the model
model, cv = load_model()

# Streamlit UI
st.title('Customer Review Sentiment Analysis')

review_input = st.text_area('Enter your review here:')
if st.button('Analyze'):
    if review_input:
        prediction = predict_sentiment(review_input, model, cv)
        st.write(f'Predicted Sentiment: {"Positive" if prediction == 1 else "Negative"}')
    else:
        st.warning('Please enter a review.')

