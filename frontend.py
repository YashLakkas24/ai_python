import streamlit as st
from app import load_pdf, chunk_text, build_index, ask_question

st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📄 AI Document Assistant")
st.write("Upload a document and ask questions about it")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Document uploaded successfully!")

    # Load and process
    with st.spinner("Processing document..."):
        text = load_pdf("temp.pdf")
        chunks = chunk_text(text)
        store = build_index(chunks)

    st.session_state["store"] = store
    st.success("Ready! Ask your questions below 👇")


# -------- CHAT UI --------
if "store" in st.session_state:
    question = st.text_input("Ask a question")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = ask_question(st.session_state["store"], question)

        st.markdown("### 🤖 Answer")
        st.write(answer)
    