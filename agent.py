from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config.models import get_model
from config.settings import THREAD_CONFIG
from prompts import load_system_prompt
from tools import get_weather, send_to_codex

agent = create_agent(
    model=get_model(), #model options and their comments are in config/models.py
    tools=[get_weather, send_to_codex],
    system_prompt=load_system_prompt(),
    checkpointer=InMemorySaver(), #conversation memory while this script is running
)

# Reuse the same thread_id so each message belongs to this conversation.
config = THREAD_CONFIG

if __name__ == "__main__":
    print("Agent started. Type 'exit' or 'quit' to leave.")

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            # Send only the new message; the checkpointer keeps previous turns.
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            print(f"Agent: {result['messages'][-1].text}")
    except (EOFError, KeyboardInterrupt):
        print()
