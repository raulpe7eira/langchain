import os

from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from operator import itemgetter

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=os.getenv("OPENAI_API_KEY")
)

class Destination(BaseModel):
  city = Field("cidade a visitar")
  reason = Field("motivo pelo qual é interessante")

parser = JsonOutputParser(pydantic_object=Destination)

city_template = PromptTemplate(
  template="Sugira uma cidade dado meu interesse por {interesting}. {output_format}",
  input_variables=["interesting"],
  partial_variables={"output_format": parser.get_format_instructions()}
)

restaurants_template = ChatPromptTemplate.from_template(
  "Sugira restaurantes populares entre locais em {city}."
)

cultures_template = ChatPromptTemplate.from_template(
  "Sugira atividades e locais culturais em {city}."
)

join_template = ChatPromptTemplate.from_messages([
  ("ai", "Sugestão de viagem para a cidade: {city}."),
  ("ai", "Restaurantes que você não pode perder: {restaurants}."),
  ("ai", "Atividades e locais culturais recomendados: {cultures}."),
  ("system", "Combine as informações anteriores em 2 parágrafos coerentes.")
])

part1 = city_template | llm | parser
part2 = restaurants_template | llm | StrOutputParser()
part3 = cultures_template | llm | StrOutputParser()
part4 = join_template | llm |  StrOutputParser()

chain = part1 | {
  "city": itemgetter("city"),
  "restaurants": part2,
  "cultures": part3
} | part4

response = chain.invoke({"interesting" : "praias" })
print(response.content)
