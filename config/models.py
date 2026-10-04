from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import settings #load the project's .env before configuring a model


#model_name = "qwen3.5:9b-q4_K_M"
#model_name = "qwen3.5:2b-q4_K_M"
model_name = "gemma4:e4b-it-q4_K_M"


def get_model():
    # Uncomment the model you want to use and comment out the other return.
    return ChatOllama(model=model_name) #local model
    #return ChatOpenAI(model="gpt-6-luna", reasoning_effort="none") #OpenAI api call model
