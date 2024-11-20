import os

from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=os.getenv("OPENAI_API_KEY")
)

loaders = [
  PyPDFLoader("GTB_standard_Nov23.pdf"),
  PyPDFLoader("GTB_gold_Nov23.pdf"),
  PyPDFLoader("GTB_platinum_Nov23.pdf")
]
documents = []
for loader in loaders:
  documents.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

answer = "Como devo proceder caso tenha um item comprado roubado"
response = qa_chain.invoke({ "query" : answer})
print(response)
