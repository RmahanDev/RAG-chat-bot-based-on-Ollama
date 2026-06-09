# RAG-chat-bot-based-on-Ollama
# Offline RAG Chatbot

A fully offline Retrieval-Augmented Generation (RAG) chatbot built with Python, FastAPI, ChromaDB, Ollama and LangChain.

## Features

* Fully offline execution
* Local LLM using Ollama
* Vector database with ChromaDB
* PDF document ingestion
* FastAPI backend
* Simple frontend
* Hallucination and relevance scoring
* RAG-based document retrieval

---

# Project Architecture

```text
Documents (PDFs)
        │
        ▼
Text Chunking
        │
        ▼
Embeddings
        │
        ▼
ChromaDB Vector Store
        │
        ▼
Retriever
        │
        ▼
Ollama LLM
        │
        ▼
Generated Answer
```

---

# Requirements

* Python 3.10+
* Ollama
* Windows / Linux / macOS

---

# Installation

## 1. Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd offline-rag-chatbot
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### PowerShell

```bash
.\venv\Scripts\Activate.ps1
```

### CMD

```bash
venv\Scripts\activate.bat
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

If Ollama is not installed:

```bash
winget install Ollama.Ollama
```

Start Ollama:

```bash
ollama serve
```

Download the model:

```bash
ollama pull qwen2.5:1.7b
```

Test the model:

```bash
ollama run qwen2.5:1.7b
```

---

# Add Documents

Place your PDF files inside the project documents folder.

Example:

```text
backend/
└── data/
    └── docs/
        ├── document1.pdf
        ├── document2.pdf
```

---

# Create Vector Database

Run:

```bash
python ingest.py
```

This will:

* Read PDF files
* Split text into chunks
* Generate embeddings
* Store vectors inside ChromaDB

---

# Run Backend

```bash
uvicorn app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

# Run Frontend

```bash
python -m http.server 5500
```

Frontend URL:

```text
http://127.0.0.1:5500
```

---

# API Example

Endpoint:

```http
POST /chat
```

Request:

```json
{
    "message": "What is Retrieval-Augmented Generation?"
}
```

Response:

```json
{
    "answer": "...",
    "score": 8.9,
    "time": 2.1,
    "hallucination_risk": 3.2
}
```

---

# Technologies Used

* Python
* FastAPI
* LangChain
* ChromaDB
* Ollama
* Qwen
* RAG
* Vector Embeddings

---

# Research Background

This project is inspired by the paper:

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**

Lewis et al., Facebook AI Research, 2020

https://arxiv.org/abs/2005.11401

---

# Author

Mahan Rezaei
Computer Science Student

