import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv() 

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


#model_name = "qwen3.5:9b-q4_K_M"
model_name = "qwen3.5:2b-q4_K_M"

agent = create_agent(
    model=ChatOllama(model=model_name), #local model
    #model=ChatOpenAI(model="gpt-6-luna", reasoning_effort="none"), #OpenAI api call model
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
