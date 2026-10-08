# Medical AI Appointment Assistant

A Generative AI-based chatbot that helps users find information about doctors, hospitals, specialties, and appointment availability.

## Features

* Doctor information
* Hospital information
* Specialty search
* Appointment availability
* RAG-based information retrieval
* Text embeddings
* FAISS vector search
* LLM-generated responses

## Technologies

* Python
* Pandas
* Sentence Transformers
* FAISS
* Google Gemini
* NumPy

## Project Flow

```text
Excel Data
    ↓
Data Processing
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retrieval
    ↓
LLM
    ↓
Final Answer
```

## Project Structure

```text
medical-ai-assistant/
│
├── data/
├── app/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── llm.py
│   └── final_answer.py
│
├── requirements.txt
└── README.md
```

## Status

✅ **Completed**

A complete Generative AI chatbot pipeline using **RAG, embeddings, FAISS, and an LLM**.
