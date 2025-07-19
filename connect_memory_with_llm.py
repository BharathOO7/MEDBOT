# 1. Make sure both classes are imported
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# 2. This is the fully corrected function
def load_llm(huggingface_repo_id):
    """
    Loads the HuggingFaceEndpoint and wraps it with ChatHuggingFace
    to handle the 'conversational' task requirement.
    """
    # Create the standard LLM endpoint
    endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
    )

    # Wrap the standard LLM with the ChatHuggingFace class
    llm = ChatHuggingFace(llm=endpoint)
    
    return llm

# The rest of your code remains exactly the same
# ===============================================

# A more robust prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If the context doesn't contain the answer, state that you cannot find relevant information in the provided documents. Do not make up an answer.

Context: {context}
Question: {question}

Provide a direct answer based on the context.
IMPORTANT: Conclude your answer with the following disclaimer: 'Disclaimer: This information is for educational purposes only. Always consult with a qualified medical professional for diagnosis and treatment.'
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

import re

def clean_text(text):
    """Removes known headers, footers, and extra whitespace."""
    # This regex removes lines like "GALE ENCYCLOPEDIA OF MEDICINE 2638"
    text = re.sub(r'GALE ENCYCLOPEDIA OF MEDICINE \d+', '', text)
    # This collapses multiple newlines into one
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

# When processing your document chunks:
# page_content = "..."
# cleaned_content = clean_text(page_content)
# Now use 'cleaned_content' to build your vector store.


# --- With This ---
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# Print the generated answer
print("✅ RESULT:")
print(response["result"])
print("\n" + "="*80 + "\n") # Add a separator

# Print the sources in a readable format
print("📚 SOURCES:")
for doc in response["source_documents"]:
    # Access metadata to show where the info came from
    source_file = doc.metadata.get('source', 'Unknown file')
    page_number = doc.metadata.get('page', 'N/A')
    print(f"  - From: {source_file}, Page: {page_number}")
    # Print the actual content chunk that was used
    # The .strip() method cleans up leading/trailing whitespace
    print(f"    Content: \"{doc.page_content.strip()}\"")
    print("-" * 20)