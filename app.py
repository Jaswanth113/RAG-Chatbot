import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

#1 Scraper
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_website_content(url: str) -> str | None:
    if not is_valid_url(url):
        return None
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print("Error:", e)
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    body = soup.find("body")
    if not body:
        return ""
    text = body.get_text(separator="\n")
    return " ".join(line.strip() for line in text.split("\n") if line.strip())

#2 Embeddings + FAISS
class EmbeddingsManager:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None

    def create_vector_store(self, text):
        doc = Document(page_content=text, metadata={"source": "website"})
        docs = self.splitter.split_documents([doc])
        self.vector_store = FAISS.from_documents(docs, self.embedder)
        return self.vector_store

#3 RAG Pipeline
class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.embeddings = EmbeddingsManager()
        self.llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
        self.vector_store = None

    def ingest(self, text):
        self.vector_store = self.embeddings.create_vector_store(text)

    def ask(self, question: str) -> dict:
        if not self.vector_store:
            return {"result": "Process a website first."}
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        prompt_template = """
        Use the following context to answer the question.
        If not in context, say "I don't know".
        Context: {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(template=prompt_template,input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
        return qa_chain.invoke({"query": question})

#4 Streamlit App
load_dotenv()
st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY in .env")
    st.stop()
    
if "rag_pipeline" not in st.session_state:
    with st.spinner("Loading AI models, please wait..."):
        st.session_state["rag_pipeline"] = RAGPipeline(groq_api_key=groq_api_key)
    st.success("Models loaded! You can now scrape a website.")

if "rag" not in st.session_state:
    st.session_state["rag"] = RAGPipeline(groq_api_key)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

url = st.text_input("Enter website URL:")
if st.button("Scrape Website"):
    text = get_website_content(url)
    if text:
        st.session_state["rag"].ingest(text)
        st.success("Website scraped successfully!")
    else:
        st.error("Failed to scrape website.")

if question := st.chat_input("Ask a question about the website..."):
    st.session_state["chat_history"].append({"role": "user", "content": question})
    response = st.session_state["rag"].ask(question)
    st.session_state["chat_history"].append({"role": "assistant", "content": response["result"]})

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])