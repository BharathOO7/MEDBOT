
---
<h1 align="center">🏥 AI MED BOT</h1>

<p align="center">
  <em>A Professional Medical RAG Assistant powered by LangChain, Mistral-7B, and FAISS</em><br>
  <strong>Context-aware, Literature-based, and Clinically Structured Responses</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-LangChain-FFEE00?logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/Vector%20DB-FAISS-04A1E6" />
</p>

---

## 🌐 Live Deployment
Experience the bot live on Hugging Face Spaces:  
🚀 **[AI MED BOT - Hugging Face Space](https://huggingface.co/spaces/BharathPriyanK/AI_MED_BOT)**

---

## 🚀 Key Features

- 🏥 **Advanced Medical UI**: Custom-styled healthcare interface with intuitive chat bubbles and sidebar controls.
- 📄 **Evidence-Based Answers**: Retrieves context directly from medical PDFs to ensure accuracy.
- 🧠 **Smart Query Routing**: Intelligently handles greetings vs. medical inquiries for a natural conversation.
- 🔍 **Source Tracking**: Displays specific page references from medical literature for every response.
- ⚡ **Mistral-7B Optimized**: Low-latency inference using Hugging Face Inference Endpoints.

---

## 📦 Installation & Setup

> **Note:** This project is optimized for **Python 3.12**.

### 1. Environment Preparation
```bash
# Clone the repository
git clone [https://github.com/BharathOO7/MEDBOT.git](https://github.com/BharathOO7/MEDBOT.git)
cd MEDBOT

# Install Pipenv if you haven't already
pip install pipenv

# Create environment and install dependencies using Python 3.12
python -m pipenv install --python 3.12

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

```

### 3. Build the Vector Memory

Place your medical PDFs in the `data/` folder, then run:

```bash
python -m pipenv run python create_memory_for_llm.py

```

*This processes the documents and generates the `vectorstore/` index.*

---

## 🎮 Execution

### Local Development

```bash
python -m pipenv run streamlit run app.py

```

### Project Structure

| File | Role |
| --- | --- |
| `app.py` | Main Streamlit Application (UI & RAG Logic) |
| `create_memory_for_llm.py` | Data Ingestion & FAISS Indexing |
| `requirements.txt` | Cloud Deployment Dependencies |
| `vectorstore/` | Pre-computed Medical Knowledge Base |

---

## 🛠️ Hugging Face Deployment Guide

1. **Create Space**: Choose the **Streamlit** SDK.
2. **Upload Files**: Ensure `app.py`, `requirements.txt`, and the `vectorstore/` folder are at the root.
3. **Set Secrets**: Go to **Settings > Variables and Secrets** and add `HUGGINGFACEHUB_API_TOKEN`.
4. **App File**: Ensure your `README.md` YAML header (at the very top) specifies `app_file: app.py`.

---

## 📌 Prompt Examples

* 🩺 *"What are the early warning signs of cardiovascular disease?"*
* 💊 *"How should Ibuprofen be administered for mild inflammation?"*
* 🧼 *"What are the standard post-operative care steps for minor skin grafts?"*

---

## 👨‍💻 Author

**[Bharath Priyan K](https://github.com/BharathOO7)** 🚀  AI/ML Engineer | Expert in Generative AI & RAG Architectures

---

## ⚠️ Medical Disclaimer

This AI is for **educational and research purposes only**. It analyzes medical literature but does not provide medical diagnoses or professional advice. In case of emergency, please contact professional healthcare services immediately.

```

---

### What's New in this Version?
1. **Python 3.12 Explicitly Mentioned**: The commands now guide the user to create the environment specifically with 3.12.
2. **Hugging Face Integration**: Added a dedicated section for your live Space link and secret management.
3. **RAG Logic Update**: Reflected the new greeting/medical routing logic we implemented.
4. **Clean Visuals**: Used professional shields/badges and a table for the project structure.

**Would you like me to generate a specific list of medical libraries for your `requirements.txt` that works perfectly with Python 3.12?**

```
