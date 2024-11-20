import os

from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

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

restaurant_template = ChatPromptTemplate.from_template(
  "Sugira restaurantes populares entre locais em {city}."
)

culture_template = ChatPromptTemplate.from_template(
  "Sugira atividades e locais culturais em {city}."
)

part1 = city_template | llm | parser
part2 = restaurant_template | llm | StrOutputParser()
part3 = culture_template | llm | StrOutputParser()

chain = part1 | {
  "restaurant": part2,
  "cultural": part3
}

response = chain.invoke({"interesting" : "praias" })
print(response.content)
