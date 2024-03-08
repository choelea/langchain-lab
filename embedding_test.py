import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#创建openai的embeddings
openai_embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
vector = FAISS.from_texts(
    ["青蛙喜欢晚上出来活动",
    "青蛙是食草动物",
     "人是由恐龙进化而来的。",
     "熊猫喜欢吃天鹅肉。",
     "1+1=5",
     "2+2=8",
     "3+3=9",
     "Gemini Pro is a Large Language Model was made by GoogleDeepMind",
     "A Language model is trained by predicting the next token"
    ],
    openai_embeddings 
)

retriever = vector.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("1+1")
print(docs[0].page_content)

docs = retriever.get_relevant_documents("青蛙的食性")
print(docs[0].page_content)

docs = retriever.get_relevant_documents("熊猫")
print(docs[0].page_content)

docs = retriever.get_relevant_documents("2加上2等于多少")
print(docs[0].page_content)

# docs = retriever.get_relevant_documents("Gemini")
# print(docs[0].page_content)

# docs = retriever.get_relevant_documents("Token")
# print(docs[0].page_content)
