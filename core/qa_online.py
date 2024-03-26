from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain

import config

llm = OpenAI(openai_api_key=config.openai_api_key)

loader = WebBaseLoader("https://fastapi.tiangolo.com/zh/deployment/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "部署是什么意思?"})
print(response["answer"])
