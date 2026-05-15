import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stTextInput>div>div>input {
    background-color: #1E1E1E;
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
}

.chat-user {
    padding: 15px;
    border-radius: 10px;
    background-color: #1E88E5;
    margin-bottom: 10px;
}

.chat-bot {
    padding: 15px;
    border-radius: 10px;
    background-color: #43A047;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("📚 Study Buddy RAG")

    st.markdown("---")

    st.subheader("Upload Study Material")

    pdf_docs = st.file_uploader(
        "Upload PDF Files",
        accept_multiple_files=True
    )

    process = st.button("Process PDFs")

    st.markdown("---")

    st.write("Built with:")
    st.write("✅ Gemini API")
    st.write("✅ FAISS")
    st.write("✅ LangChain")
    st.write("✅ Streamlit")

# =========================
# PDF TEXT EXTRACTION
# =========================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            text += page.extract_text()

    return text

# =========================
# TEXT CHUNKING
# =========================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return chunks

# =========================
# VECTOR STORE
# =========================

def get_vector_store(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("vector_store")

# =========================
# CONVERSATIONAL CHAIN
# =========================

def get_conversational_chain():

    prompt_template = """

    Answer the question only from the provided context.

    If answer is not available in the context,
    say:
    "Answer is not available in the uploaded documents."

    Context:
    {context}

    Question:
    {question}

    Answer:

    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain

# =========================
# USER QUESTION HANDLER
# =========================

def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    new_db = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    answer = response["output_text"]

    # CHAT HISTORY
    st.session_state.chat_history.append(
        {
            "question": user_question,
            "answer": answer
        }
    )

    # DISPLAY CHAT
    for chat in st.session_state.chat_history:

        st.markdown(
            f"""
            <div class="chat-user">
            👨‍🎓 <b>You:</b><br>{chat['question']}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="chat-bot">
            🤖 <b>Study Buddy:</b><br>{chat['answer']}
            </div>
            """,
            unsafe_allow_html=True
        )

    # SOURCE CHUNKS
    st.subheader("📄 Retrieved Source Chunks")

    for i, doc in enumerate(docs):

        with st.expander(f"Chunk {i+1}"):

            st.write(doc.page_content)

# =========================
# PROCESS PDFs
# =========================

if process:

    if pdf_docs:

        with st.spinner("Processing PDFs..."):

            raw_text = get_pdf_text(pdf_docs)

            text_chunks = get_text_chunks(raw_text)

            get_vector_store(text_chunks)

            st.success("PDFs Processed Successfully!")

    else:
        st.warning("Please Upload PDF Files")

# =========================
# MAIN CHAT AREA
# =========================

st.header("💬 Ask Questions From Your Notes")

user_question = st.text_input(
    "Ask a Question"
)

if user_question:

    user_input(user_question)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <center>
    Made with ❤️ using Gemini + FAISS + Streamlit
    </center>
    """,
    unsafe_allow_html=True
)

