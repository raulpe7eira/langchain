import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

number_of_days = 7
number_of_kids = 2
activity = "praia"

prompt_template = PromptTemplate.from_template(
  "Crie um roteiro de viagem de {days} dias, para uma família com {kids} crianças, que gostam de {activity}."
)

prompt = prompt_template.format(
  days=number_of_days,
  kids=number_of_kids,
  activity=activity
)
print(prompt)

llm = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=os.getenv("OPENAI_API_KEY")
)

response = llm.invoke(prompt)
print(response.content)
