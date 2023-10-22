import streamlit as st
import joblib

# Load the vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load the model
model = joblib.load("best_model.pkl")

# Create a title
st.title("Predict Fake News")

# Input text
text = st.text_input("Enter the news article:")

# Button to run inference
if st.button("Predict"):

    # Convert the text to a vector
    vector = vectorizer.transform([text])

    # Predict the label
    label = model.predict(vector)[0]

    # Display the label
    if label == 0:
        st.write("The news article is fake.")
    else:
        st.write("The news article is real.")

# Add a link to BBC News
st.markdown("[BBC News](https://www.bbc.com/news)")
st.markdown("[ITV News](https://www.itv.com/news)")
