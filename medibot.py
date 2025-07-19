import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Caching Functions for Efficiency ---

@st.cache_resource
def get_vectorstore():
    """Loads the vector store from the local path. Caches the result for performance."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

# --- THIS FUNCTION HAS BEEN UPDATED WITH A MORE DETAILED PROMPT ---
def get_custom_prompt():
    """
    Defines and returns a more sophisticated prompt template to guide the LLM
    in generating a clear, structured, and comprehensive answer.
    """
    CUSTOM_PROMPT_TEMPLATE = """
    You are an AI medical assistant. Your task is to answer the user's question based on the provided context documents.
    Synthesize the information from all context chunks into a single, well-structured, and easy-to-read answer. Do not just summarize each document.

    Follow these steps:
    1.  First, directly answer the user's question.
    2.  Then, provide a more detailed explanation using the following structure:
        -   **Primary Treatments:** Detail the main medical treatments mentioned (e.g., surgery, chemotherapy, radiation). Explain their purpose based on the context.
        -   **Complementary Therapies:** Describe any alternative or complementary treatments mentioned that help with well-being or side effects.
    3.  If the context does not contain the answer, state that the information is not available in the provided documents. Do not make up information.

    Context: {context}
    Question: {question}

    Your Answer:
    IMPORTANT: Conclude your entire response with the following disclaimer: 'Disclaimer: This information is for educational purposes only. Always consult with a qualified medical professional for diagnosis and treatment.'
    """
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    return prompt

@st.cache_resource
def load_qa_chain():
    """
    Loads and caches the entire RetrievalQA chain, including the LLM and retriever.
    This is a major performance enhancement.
    """
    try:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            st.error("Hugging Face API token not found. Please set the HF_TOKEN environment variable.")
            return None

        endpoint = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.5,
            max_new_tokens=512,
        )
        llm = ChatHuggingFace(llm=endpoint)
        vectorstore = get_vectorstore()
        prompt_template = get_custom_prompt()

        if vectorstore is None:
            return None

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt_template}
        )
        return qa_chain

    except Exception as e:
        st.error(f"Failed to load the QA chain: {e}")
        return None

# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="MedBot", page_icon="�")
    st.title("🩺 MedBot")
    st.write("Your personal AI medical assistant. Ask me anything from the medical encyclopedia.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    qa_chain = load_qa_chain()

    if prompt := st.chat_input("Ask a question about medicine..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        greetings = ["hi", "hello", "hey", "hallo", "yo"]
        if prompt.lower().strip() in greetings:
            response = "Hello! I'm MedBot. How can I help you with your medical questions today?"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        else:
            if qa_chain is None:
                st.error("The application is not configured correctly. Please check the logs.")
                return

            with st.spinner("Thinking..."):
                try:
                    response = qa_chain.invoke({'query': prompt})
                    result = response.get("result", "Sorry, I encountered an error.")
                    source_documents = response.get("source_documents", [])

                    with st.chat_message("assistant"):
                        st.markdown(result)
                        with st.expander("View Sources"):
                            for doc in source_documents:
                                source_file = doc.metadata.get('source', 'Unknown')
                                page_number = doc.metadata.get('page', 'N/A')
                                content = doc.page_content
                                st.write(f"**Source:** {os.path.basename(source_file)}, **Page:** {page_number}")
                                st.text_area("Content:", value=content, height=150, disabled=True)

                    st.session_state.messages.append({"role": "assistant", "content": result})

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()