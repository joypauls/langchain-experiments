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


loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

chain = create_stuff_documents_chain(llm, prompt)

result = chain.invoke({"context": docs})
print("\nSUMMARY:\n")
print(result)
print("\n")
