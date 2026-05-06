import streamlit as st
import requests

st.title("PDF app")

uploaded_file = st.file_uploader("Upload")
if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post("http://127.0.0.1:8000/files", files=files)
    st.success(f"Uploaded Successfully")

question = st.text_input("Ask a question")
if st.button("Ask"):
    response = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
    st.write(response.json()["answer"])
