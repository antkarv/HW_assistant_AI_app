<<<<<<< HEAD
---
title: HW Assistant AI App
emoji: 💻
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
short_description: Hardware QA Assistant is an advanced AI system
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
# 🔧 Hardware QA Assistant – Advanced LangGraph AI System

> AI-powered Hardware Design & Datasheet Analysis Assistant  
> Built with LangGraph, FastAPI, Gradio, ChromaDB, Vision LLM & JWT Authentication

---

## 🚀 Overview

**Hardware QA Assistant** is an advanced AI system designed to:

- Analyze hardware design questions
- Parse and index datasheets (PDF / DOCX)
- Extract structured information from schematic images (Vision LLM)
- Perform multi-query semantic retrieval (RAG v2 pipeline)
- Rerank results using Cross-Encoder
- Optionally enrich answers with live web search
- Provide execution trace & LLM inspection
- Support secure user authentication (JWT + SQLite)

This project demonstrates a **production-style AI architecture**, not just a simple chatbot.

---

## 🧠 Architecture

The system is built around a LangGraph state machine:

```
session → router → ingest → multiquery
        → retrieve_pool → rrf_fuse → rerank → mmr
        → parent_promote → (web?) → answer
        → verify → clarify | END
```

### Key Capabilities

- ✔ Multi-query retrieval
- ✔ Reciprocal Rank Fusion (RRF)
- ✔ Cross-encoder reranking
- ✔ MMR diversification
- ✔ Vision-based schematic extraction
- ✔ Parent chunk promotion
- ✔ Execution trace inspector
- ✔ LLM token usage tracking
- ✔ JWT-based authentication
- ✔ Persistent vector database (Chroma)

Architecture diagram:

![LangGraph Flow](langgraph-visualization.png)

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Groq (OpenAI-compatible API) |
| Orchestration | LangGraph |
| Vector DB | Chroma (Persistent) |
| Embeddings | Sentence-Transformers |
| Reranker | Cross-Encoder (MiniLM) |
| OCR | PaddleOCR / Tesseract |
| Vision | LLaMA Vision model |
| Web Search | Tavily |
| Backend | FastAPI |
| UI | Gradio |
| Auth | JWT + SQLite |
| Deployment | Uvicorn |

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/HW_assistant_AI_app.git
cd HW_assistant_AI_app
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# or
source .venv/bin/activate   # Linux / macOS
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Create `.env` File

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
AUTH_SECRET=your_super_secret
TAVILY_API_KEY=optional
```

---

### 5️⃣ Create First User

```bash
python -c "from users import create_user; create_user('admin@example.com','StrongPass!23', True)"
```

---

### 6️⃣ Run the Application

```bash
uvicorn secure_server:app --host 0.0.0.0 --port 8000
```

Open in browser:

```
http://localhost:8000
```

---

## 🔍 Feature Breakdown

### 📄 PDF / DOCX Ingestion
- Page-first extraction
- OCR fallback when needed
- Hybrid chunking
- Persistent Chroma storage

### 🖼️ Schematic Understanding
- Vision LLM extracts structured JSON
- Components & nets detected
- Datasheet expansion via RAG

### 🔎 Advanced Retrieval Pipeline
- Multi-query expansion
- RRF fusion
- Cross-encoder reranking
- MMR selection
- Parent chunk promotion

### 🧪 LLM Inspector
- Token usage tracking
- Prompt preview
- Latency metrics
- Execution trace
- DAG visualization

### 🔐 Authentication
- JWT-based sessions
- SQLite user store
- Protected `/app` routes
- Secure logout

---

## 📊 Example Use Cases

- Bias current calculation from datasheet
- LDO stability analysis
- Pin mapping verification
- Schematic functional explanation
- Datasheet parameter extraction
- Hardware design validation

---

## 🛠️ Project Structure

```
.
├── hardware_qa_assistant_lang_graph_gradio_demo_version_users.py
├── hardware_qa_assistant_helpers.py
├── secure_server.py
├── users.py
├── requirements.txt
├── langgraph-visualization.png
└── README.md
```

---

## 🔒 Security Notes

- `.env` is excluded via `.gitignore`
- Vector DB is local-only
- SQLite user DB is not versioned
- JWT tokens include expiration
- No API keys are stored in repository

---

## 📈 Future Improvements

- Docker containerization
- Production-ready deployment configuration
- Redis-based session memory
- Multi-tenant workspace isolation
- CI/CD pipeline
- HuggingFace / Render deployment

---

## 👨‍💻 Author

**Antonios Karvelas**  
AI Systems Engineer | Telecom Architect  

---
>>>>>>> 6254b0c (Add README.md for Hardware QA Assistant project)
