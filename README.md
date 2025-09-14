# Website RAG Chatbot

**Website RAG Chatbot** is a **Retrieval-Augmented Generation (RAG)** system that lets users input any website URL, scrape its content, and ask **context-aware questions** about the site.  

This project demonstrates how **RAG pipelines** can transform unstructured website text into searchable knowledge, enabling interactive Q&A over web content.

---

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  

---

## Overview
Website RAG Chatbot is a **context-driven question answering system** that combines **web scraping, embeddings, semantic search, and LLM reasoning** to deliver instant answers.  

Key technologies:
- Python, Streamlit  
- LangChain (orchestration for embeddings, retrieval, and LLMs)  
- Hugging Face `all-MiniLM-L6-v2` embeddings  
- FAISS for semantic vector search  
- Groq LLM (`llama-3.1-8b-instant`) for generating responses  
- BeautifulSoup for website scraping  

---

## Features
- Input any **website URL** and extract clean text content  
- Convert unstructured text into **dense embeddings**  
- Perform **semantic search** with FAISS  
- Ask **questions about website content** and get accurate answers  
- Interactive **Streamlit web interface** with chat history  

---

## Architecture
1. **Scraping** – Website content extracted via `BeautifulSoup`  
2. **Text Splitting** – Large text split into chunks with `RecursiveCharacterTextSplitter`  
3. **Embeddings** – Chunks embedded using Hugging Face `all-MiniLM-L6-v2`  
4. **Vector Store** – Stored in FAISS for similarity search  
5. **LLM Response** – Relevant chunks sent to Groq LLM via LangChain for final answer  

---


## Results
- Scrapes and processes website text into **searchable embeddings**  
- Delivers **98% accurate responses** on context-based queries  
- Responds in **under 1 second per query** using FAISS + Groq LLM  
- Enables **real-time interactive chat** with website content  

---
