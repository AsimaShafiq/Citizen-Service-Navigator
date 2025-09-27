# agent_b.py

import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import streamlit as st

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# LLM Setup
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------
# Embedding Model
# ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Load FAISS index
# ---------------------------
faiss_index = FAISS.load_local("policy_index", embedder, allow_dangerous_deserialization=True)

# ---------------------------
# Prompt Template for Eligibility
# ---------------------------
eligibility_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
Citizen query: {query}

Relevant policies:
{context}

You are Agent B (Eligibility Evaluator).
Tasks:
1. Decide clearly if the citizen is ELIGIBLE or NOT ELIGIBLE.
2. Explain reasoning in plain, local-style language.
3. If eligible, suggest next step (like form filling, required docs).
4. If information is incomplete, mention assumptions you made.
"""
)

eligibility_chain = LLMChain(llm=llm, prompt=eligibility_prompt)

# ---------------------------
# Function: Run Agent B
# ---------------------------
def check_eligibility(user_query: str):
    docs = faiss_index.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = eligibility_chain.run(query=user_query, context=context)
    return result

# ---------------------------
# Streamlit UI
# ---------------------------
def run_ui():
    st.title("Citizen Service Navigator - Agent B (Eligibility Checker)")

    user_query = st.text_input("Ask your question (e.g., 'Am I eligible for housing support?')")

    if st.button("Check Eligibility"):
        if user_query.strip():
            with st.spinner("Evaluating..."):
                response = check_eligibility(user_query)
                st.subheader("Eligibility Decision")
                st.write(response)

if _name_ == "_main_":
    run_ui()
