from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

llm = OpenAI(openai_api_key=config.openai_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能研究方面的专家，精通大模型语言的相关知识，也会进行Python编程."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser


output = chain.invoke({"input": "神经网络是什么?"})

print(output)
