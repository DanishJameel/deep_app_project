import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('stopwords')
model = pickle.load(open('model_dump.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction

def main():
    st.title("Fake News Detection With Deep Learning and NLP Techniques")

    # Text input
    text = st.text_input("Enter the News to Predict", "")

    if st.button("Classify"):
        prediction = predict(text)
        st.write(f"The News is {prediction} news")

if __name__ == "__main__":
    main()
