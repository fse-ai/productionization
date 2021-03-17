import requests

import streamlit as st


news_labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tec"
}

st.title('Text classification')
st.write("""Analyze text and classify them into four""")

text = st.text_input("Enter text here")
if st.button("Predict"):
    response = requests.get(f"http://127.0.0.1:8000/classify/{text}")
    data_dict = response.json()
    print(data_dict)
    prediction = data_dict["prediction"]
    confidence = data_dict["confidence"]
    st.success(f"This text is classified as {news_labels[prediction]} "
               f"with {round(confidence * 100, 2)} % confidence")
