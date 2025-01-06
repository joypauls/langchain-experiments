from dotenv import load_dotenv
import os

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise Exception("Please set the OPENAI_API_KEY environment variable.")

from dotenv import load_dotenv
import textwrap
import bs4
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback
import streamlit as st
import tempfile


st.set_page_config(page_title="PDF Assistant", layout="centered")
st.title("PDF Assistant")

st.sidebar.title("Choose a PDF")
# st.sidebar.write(
#     "Upload a PDF file to generate a structured summary using LangChain and OpenAI. "
#     "Make sure your OpenAI API key is set as an environment variable (OPENAI_API_KEY)."
# )

access_type = st.sidebar.radio(
    "Access Type",
    ["URL", "Upload"],
    label_visibility="hidden",
)

pdf_loader_input = None
if access_type == "URL":
    pdf_loader_input = st.sidebar.text_input("Enter a valid URL")
elif access_type == "Upload":
    uploaded_file = st.sidebar.file_uploader("File Upload", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            pdf_loader_input = temp_pdf.name


submit = st.sidebar.button("Submit", type="primary")
if submit:

    loader = PyPDFLoader(pdf_loader_input)
    documents = loader.load()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Read the following and provide a concise summary:\\n\\n{context}",
            )
        ]
    )
    chain = create_stuff_documents_chain(llm, prompt)

    with get_openai_callback() as cb:
        result = chain.invoke({"context": documents})
        summary = result

    st.write("### Summary:")
    st.write(summary)
