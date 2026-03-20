# 🧬 MediRAG AI — Medical Q&A System using RAG + Endee

A production-grade medical AI assistant built using Retrieval Augmented Generation with Endee as the vector database and LLaMA 3 as the language model.

## 🔗 Live Demo
👉 [Click here to open the app](https://medirag-ai-4v4miutwq4nir8rrodapbw.streamlit.app)

## 📌 Project Overview
MediRAG AI lets anyone ask medical questions in 10 languages and get accurate, detailed answers instantly. It includes a symptom checker, BMI calculator, medicine information lookup, and emergency contacts — all powered by AI.

## ❓ Problem Statement
Millions of people lack access to quick, reliable medical information. Language barriers and lack of medical knowledge make it worse. MediRAG AI bridges this gap by providing instant AI-powered medical answers grounded in verified knowledge, in the user's own language.

## ⚙️ System Design & Technical Approach
```
User Question
     ↓
SentenceTransformer Embedding (all-MiniLM-L6-v2)
     ↓
Endee Vector Database — Semantic Search
     ↓
Top 3 Relevant Medical Documents Retrieved
     ↓
LLaMA 3.3 70B via Groq API
     ↓
Final Accurate Answer Displayed
```

## 🗄️ How Endee is Used
Endee serves as the core vector database in this project. Medical documents are converted into vector embeddings using SentenceTransformers and stored in Endee. When a user asks a question, it is embedded into the same vector space and Endee performs semantic similarity search to retrieve the top 3 most relevant documents. These documents are passed as context to LLaMA 3 which generates a comprehensive, grounded answer.

## ✨ Features
- 💬 Medical Q&A powered by RAG and LLaMA 3.3 70B
- 🔍 AI Symptom Checker — enter symptoms, get possible conditions
- ⚖️ BMI Calculator with personalized AI health advice
- 💊 Medicine information — dosage, uses, side effects
- 🚨 Emergency contacts and first aid guide
- 🎤 Voice input support in 10 languages
- 🌙 Dark and Light mode
- 💾 Chat history saved automatically
- 👍 👎 Feedback buttons on every answer

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Database | Endee |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 |
| Language Model | LLaMA 3.3 70B via Groq |
| Frontend | Streamlit |
| Language | Python 3.12 |

## 🚀 Setup & Execution
```bash
# Clone the repository
git clone https://github.com/velpula-manish/medirag-ai
cd medirag-ai

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key in Streamlit secrets
# Create .streamlit/secrets.toml and add:
# GROQ_API_KEY = "your_groq_api_key"

# Run the app
streamlit run app.py
```

## ✅ Mandatory Repository Steps Completed
- ⭐ Starred the official Endee repository at github.com/endee-io/endee
- 🍴 Forked the repository to personal GitHub account
- 🏗️ Built the project using Endee as the core vector database
