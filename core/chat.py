from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import config

llm = OpenAI(openai_api_key=config.openai_api_key, model="gpt-3.5-turbo", temperature=0)

loader = WebBaseLoader("https://fastapi.tiangolo.com/zh/deployment/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:

<context>
{context}
</context>

Question: {input}""")


prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "给定上述对话，生成一个搜索查询以查找与对话相关的信息")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [HumanMessage(content="可以帮助我部署FastApi吗?"), AIMessage(content="可以!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "告诉我怎么做"
})

prompt = ChatPromptTemplate.from_messages([
    ("system", "根据下面的上下文回答用户的问题:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="可以帮助我部署FastApi吗?"), AIMessage(content="可以!")]
output = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "告诉我怎么做"
})

print(output["answer"])
