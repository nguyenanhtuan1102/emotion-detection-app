import pickle
import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# loading model
log = pickle.load(open("model/log.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

# clean text
stopwords_set = set(stopwords.words('english'))
def clean_text(text):
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

def main():
    st.title("Sentiment Analysis")
    comment = st.text_input("Enter Your Comment")
    button = st.button("Analyze")

    if button:
        text = comment.title()
        cleaned = clean_text(text)
        input_feature = tfidf.transform([cleaned])
        predictions = log.predict(input_feature)[0]

        # Map category ID to category name
        emotion_labels = {
            0: "SADNESS",
            1: "JOY",
            2: "LOVE",
            3: "ANGER",
            4: "FEAR",
            5: "SURPRICE"
        }

        emotions = emotion_labels.get(predictions, "Unknown")
        st.write("Emotion: ", emotions)

if __name__ == '__main__':
    main()