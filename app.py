import streamlit as st
import requests

st.title("PDF app")

uploaded_file = st.file_uploader("Upload")
if uploaded_file:
    files = {"file": uploaded_file}

    response = requests.post("http://127.0.0.1:8000/files", files=files)
    if response.status_code == 200:
        st.success("Uploaded Successfully")
    else:
        st.error(response.json().get("detail", "Upload failed"))
question = st.text_input("Ask a question")

if st.button("Ask"):
    response = requests.post("http://127.0.0.1:8000/ask", json={"question": question})

    st.write("Status code:", response.status_code)
    st.write("Response:", response.text)
