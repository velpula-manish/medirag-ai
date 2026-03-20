# 🧬 MediRAG AI

<div align="center">

**Medical AI Assistant powered by RAG + Endee + LLaMA 3**

[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://medirag-ai-4v4miutwq4nir8rrodapbw.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/velpula-manish/medirag-ai)
[![Endee](https://img.shields.io/badge/Endee-Vector%20DB-00C853?style=for-the-badge)](https://github.com/endee-io/endee)
[![LLaMA](https://img.shields.io/badge/LLaMA%203-70B-7C3AED?style=for-the-badge)](https://groq.com)

[Live Demo](https://medirag-ai-4v4miutwq4nir8rrodapbw.streamlit.app) • [GitHub](https://github.com/velpula-manish/medirag-ai) • [Endee](https://github.com/endee-io/endee)

</div>

---

## What is MediRAG AI?

MediRAG AI is a production-grade medical assistant that answers any health question instantly in 10 languages. Built using Retrieval Augmented Generation with Endee as the vector database and LLaMA 3.3 70B as the language model.

---

## How Endee is Used

This project is built with the Endee vector database architecture. The RAG pipeline 
uses Endee's Python client (`pip install endee`) for vector storage and semantic search.

The vector embeddings are created using SentenceTransformers and stored using 
Endee's index API:

from endee import Endee
from endee.index import VectorItem

client = Endee()
idx = client.create_index(name="medical", dimension=384, space_type="cosine")
idx.upsert([VectorItem(id=str(i), vector=embedding, metadata={"text": doc})])
results = idx.query(vector=query_vector, top_k=3)

Note: Endee requires a running server instance. For the hosted deployment on 
Streamlit Cloud, FAISS is used as a compatible fallback since Endee's server 
requires Docker which is not available on Streamlit Cloud's free tier.
```
User Question → Embedding → Endee Vector Search → Top 3 Docs → LLaMA 3 → Answer
```

---

## Features

| Feature | Description |
|---------|-------------|
| 💬 Medical Q&A | Ask any medical question in 10 languages |
| 🔍 Symptom Checker | Enter symptoms, get possible conditions |
| ⚖️ BMI Calculator | Calculate BMI with AI health advice |
| 💊 Medicine Info | Dosage, uses and side effects lookup |
| 🚨 Emergency Contacts | India emergency numbers and first aid |
| 🎤 Voice Input | Speak your question in your language |
| 🌙 Dark / Light Mode | Toggle between themes |
| 💾 Chat History | Conversations saved automatically |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Database | Endee |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 |
| Language Model | LLaMA 3.3 70B via Groq |
| Frontend | Streamlit |
| Language | Python 3.12 |

---

## System Design
```
┌─────────────────────────────────────────────────────┐
│                   MediRAG AI                        │
├─────────────────────────────────────────────────────┤
│  User Question                                      │
│       ↓                                             │
│  SentenceTransformer Embedding                      │
│       ↓                                             │
│  Endee Vector Database — Semantic Search            │
│       ↓                                             │
│  Top 3 Relevant Medical Documents                   │
│       ↓                                             │
│  LLaMA 3.3 70B via Groq API                        │
│       ↓                                             │
│  Final Answer Displayed on Streamlit UI             │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start
```bash
# Clone the repository
git clone https://github.com/velpula-manish/medirag-ai
cd medirag-ai

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
# Create .streamlit/secrets.toml
# GROQ_API_KEY = "your_groq_api_key"

# Run the app
streamlit run app.py
```

---

## Mandatory Repository Steps Completed

- ⭐ Starred the official Endee repository — [endee-io/endee](https://github.com/endee-io/endee)
- 🍴 Forked the repository to personal GitHub account
- 🏗️ Built the project using Endee as the core vector database

---

## Live Demo

👉 [Click here to open MediRAG AI](https://medirag-ai-4v4miutwq4nir8rrodapbw.streamlit.app)

---

<div align="center">
Built with ❤️ by <a href="https://github.com/velpula-manish">Manishkumar</a>
</div>
