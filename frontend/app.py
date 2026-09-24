import streamlit as st
import requests

st.title("PDF QA System")

uploaded_file = st.file_uploader("Upload")

if uploaded_file:
    if st.button("Process PDF"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

    with st.spinner("Processing PDF..."):
        try:
            response = requests.post("http://127.0.0.1:8000/files", files=files)

            if response.status_code == 200:
                st.success("Uploaded Successfully")
                st.json(response.json())
            else:
                st.error(response.json().get("detail", "Upload failed"))
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
question = st.text_input("Ask a question")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask", json={"question": question}
                )
                st.write("Status Code:", response.status_code)
                st.json(response.json())
            except Exception as e:
                st.error(f"Error querying backend: {e}")
