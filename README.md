---


<h1 align="center">🧠💬 MEDBOT</h1>

<p align="center">
  <em>Conversational Medical Chatbot powered by LangChain, Hugging Face, FAISS, and Streamlit</em><br>
  <strong>A smarter way to search medical insights—context-aware, memory-enhanced, and user-friendly</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/languages/top/BharathOO7/MEDBOT?color=blue" />
  <img src="https://img.shields.io/github/last-commit/BharathOO7/MEDBOT" />
  <img src="https://img.shields.io/github/issues/BharathOO7/MEDBOT" />
  <img src="https://img.shields.io/github/stars/BharathOO7/MEDBOT?style=social" />
</p>

---

## 🚀 Features

- 🧠 **Conversational Memory**: Uses LangChain's memory module for enriched dialogues.
- 📄 **Document-Aware Answers**: Context pulled directly from PDFs using FAISS.
- 🧬 **Hugging Face LLM Integration**: Smart, generative responses tailored for medical queries.
- 💻 **Streamlit Frontend**: Clean and interactive interface for user input/output.
- 🔍 **Semantic Search**: Uses FAISS for efficient, high-relevance document matching.
- 🧠 **LLM Memory Connector**: Seamless retrieval pipeline with HuggingFace + LangChain.

---

## 📦 Installation

> Recommended: Python 3.10+ with Pipenv installed

```bash
# Clone the repository
git clone https://github.com/BharathOO7/MEDBOT.git
cd MEDBOT

# Install core dependencies
python -m pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install sentence-transformers

# Activate environment
python -m pipenv shell
```

---

## 🧠 Vector Store Creation

```bash
python -m pipenv run python create_memory_for_llm.py
```

> This will ingest your medical PDFs and create FAISS indexes.

---

## 🔌 Connect Memory with LLM

```bash
python connect_memory_with_llm.py
```

> Initializes memory-aware pipeline with document retriever and HuggingFace endpoint.

---

## 🎮 Launch Streamlit App

```bash
streamlit run medibot.py
```

> Your chatbot is now live at `localhost:8501` (or via Streamlit Community Cloud).

---

## ⚙️ Project Structure

| File | Description |
|------|-------------|
| `medibot.py` | Streamlit UI |
| `create_memory_for_llm.py` | Creates FAISS index from PDFs |
| `connect_memory_with_llm.py` | Builds LangChain pipeline |
| `.env.template` | Add your HuggingFace key here |
| `README.md` | You're reading it! |

---

## 🔐 Environment Setup

Create a `.env` file using `.env.template`:

```bash
HUGGINGFACE_API_KEY=your_token_here
```

> Never commit `.env`—it's ignored via `.gitignore`.

---

## 📌 Prompt Ideas

```text
🩺 What are the symptoms of dengue in early stages?
💊 How do I treat mild fever without medication?
🧼 Precautions after surgical stitches removal?
```

---

## 👨‍💻 Author

Built with precision and passion by **[Bharath Priyan K](https://github.com/BharathOO7)**  
🚀 Applied AI/ML Engineer at Bharath Marine Service Pvt. Ltd.

---

## 🌐 Deploy on Streamlit Cloud

1. Connect your GitHub repo: [`MEDBOT`](https://github.com/BharathOO7/MEDBOT.git)
2. Choose branch: `main`
3. App entry point: `medibot.py`
4. Done ✅

---

## 💡 Future Enhancements

- 🔄 RAG pipeline with ChatML formatting
- 📊 Analytics dashboard for user interactions
- 🗂️ Multi-source PDF ingestion & filtering
- 🧬 Fine-tuned medical LLM support

---

---

You can copy-paste this directly into your `README.md`. Want me to help polish the code snippets, deploy a live demo link, or stylize your repo homepage even more? I’m in—let’s make MEDBOT shine like a real product launch ✨
