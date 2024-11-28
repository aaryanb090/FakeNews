import streamlit as st
import pickle
import numpy as np

# Load the trained model and Vectorizer
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)


# Streamlit UI
st.title("Fake News Detection App")
st.write("Enter the details below to predict if the news is real or fake.")

# Input fields
title = st.text_input("News Title")
author = st.text_input("Author Name")
article = st.text_area("News Article")

# Predict button logic
if st.button("Predict"):
    if title and author and article:
        # Combine inputs
        input_text = f"{author} {title} {article}"

        # Handle input preprocessing
        if vectorizer:
            # If a vectorizer exists, transform the input text
            input_vector = vectorizer.transform([input_text])  # TF-IDF transformation
            prediction = model.predict(input_vector)  # Predict using the model
        else:
            # If no vectorizer, use raw text input converted to a NumPy array
            input_array = np.array([input_text])  # Convert to 2D NumPy array
            prediction = model.predict(input_array)  # Predict using the model

        # Interpret the prediction result
        result = "Fake News" if prediction[0] == 1 else "Real News"

        # Display the result
        st.success(f"The news is classified as: {result}")
    else:
        st.error("Please fill out all fields!")
