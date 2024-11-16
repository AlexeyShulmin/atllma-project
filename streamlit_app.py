import streamlit as st
import requests

st.title("Advanced search in matplotlib.axes.Axes")

query = st.text_input("Enter your query")

if st.button("Search"):
    response = requests.post('http://localhost:5000/query', json={'query': query})
    st.write(response.json()['response'])
    