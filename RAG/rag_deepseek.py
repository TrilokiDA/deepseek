# Created by trilo at 21-02-2025
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Titel and Subtitle
st.title("📘 RAG Chatbot")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:7b"],
        index=0
    )
    st.divider()
    st.markdown("### PDF Upload")
    # File Upload
    upload_pdf = st.file_uploader(
        "Upload Research Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False

    )
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)"
                " | [Streamlit](https://streamlit.io/)")

PDF_STORAGE_PATH = 'document/'
EMBEDDING_MODEL = OllamaEmbeddings(model=selected_model)
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model=selected_model)

def save_upload_file(upload_file):
    file_path = PDF_STORAGE_PATH + upload_file.name
    with open(file_path, "wb") as file:
        file.write(upload_file.getbuffer())
    return file_path


def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()


def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)


def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)


def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)


# Prompting
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
If question is not available in Knowledge Base say sorry this is not present in Database.
Always say Thanks for asking question in last. 

Query: {user_query} 
Context: {document_context} 
Answer:
"""


def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


if upload_pdf:
    save_path = save_upload_file(upload_pdf)
    raw_docs = load_pdf_documents(save_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

    st.success("✅ Document processed successfully! Ask your questions below.")

    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
