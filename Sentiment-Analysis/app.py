import streamlit as st
import pickle

# Load saved model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page title
st.title("Sentiment Analysis UI")
st.write("Enter a sentence to check whether it is Positive or Negative.")

# User input
user_input = st.text_area("Enter text here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = user_input.lower()
        text_vector = vectorizer.transform([clean_text])
        prediction = model.predict(text_vector)[0]

        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")



# To Run App Streamlit run app.py