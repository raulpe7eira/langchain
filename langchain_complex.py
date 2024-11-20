import os

from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=os.getenv("OPENAI_API_KEY")
)

city_template = ChatPromptTemplate.from_template(
  "Sugira uma cidade dado meu interesse por {interesting}. A sua saída deve ser somente o nome da cidade. Cidade: "
)

restaurant_template = ChatPromptTemplate.from_template(
  "Sugira restaurantes populares entre locais em {city}"
)

culture_template = ChatPromptTemplate.from_template(
  "Sugira atividades e locais culturais em {city}"
)

chain = SimpleSequentialChain(chains=[
    LLMChain(prompt=city_template, llm=llm),
    LLMChain(prompt=restaurant_template, llm=llm),
    LLMChain(prompt=culture_template, llm=llm)
  ],
  verbose=True
)

response = chain.invoke("praia")
print(response.content)
