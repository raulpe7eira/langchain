import os

from dotenv import load_dotenv

from langchain.chains import ConversationChain
from langchain.globals import set_debug
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=os.getenv("OPENAI_API_KEY")
)

messages = [
  "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
  "Qual é o melhor período do ano para visitar em termos de clima?",
  "Quais tipos de atividades ao ar livre estão disponíveis?",
  "Alguma sugestão de acomodação eco-friendly por lá?",
  "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
  "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes."
]

memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
  llm=llm,
  memory=memory,
  verbose=True
)

for message in messages:
  response = conversation.predict(input=message)
  print(response)
