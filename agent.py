from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config.models import get_model
from config.settings import THREAD_CONFIG
from prompts import load_system_prompt
from tools import get_weather, send_to_codex
from audio.voice import VoiceInput, configure_terminal, review_transcript
from audio.speech import VoiceOutput
from config.voice import TTS_MODELS

agent = create_agent(
    model=get_model(), #model options and their comments are in config/models.py
    tools=[get_weather, send_to_codex],
    system_prompt=load_system_prompt(),
    checkpointer=InMemorySaver(), #conversation memory while this script is running
)

# Reuse the same thread_id so each message belongs to this conversation.
config = THREAD_CONFIG

if __name__ == "__main__":
    configure_terminal()
    print("Agent started. Type /voice to dictate, /voice-test to test transcription, or 'exit' to leave.")
    voice = VoiceInput() #the model loads on first use and is then reused
    speaker = VoiceOutput()
    print("Voice replies are spoken. Switch TTS: /tts MODEL [hr|en]; list: /tts.")

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            if user_input.split()[0].lower() == "/tts":
                parts = user_input.split()
                if len(parts) == 1:
                    print(f"TTS: {speaker.name}, {speaker.language}. Models: {', '.join(TTS_MODELS)}")
                else:
                    try:
                        if len(parts) > 3:
                            raise ValueError("Usage: /tts MODEL [hr|en]")
                        speaker = VoiceOutput(parts[1], parts[2] if len(parts) == 3 else speaker.language)
                        print(f"TTS selected: {speaker.name}, {speaker.language}")
                    except ValueError as error:
                        print(error)
                continue

            spoken_reply = user_input.lower() == "/voice"
            if user_input.lower() in {"/voice", "/voice-test"}:
                test_only = user_input.lower() == "/voice-test"
                try:
                    transcript = voice.dictate()
                    if test_only:
                        continue
                    user_input = review_transcript(transcript)
                except KeyboardInterrupt:
                    print("\nVoice message cancelled.")
                    continue
                except Exception as error:
                    print(f"Voice error: {error}")
                    continue
                if not user_input:
                    continue

            # Send only the new message; the checkpointer keeps previous turns.
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            answer = result['messages'][-1].text
            print(f"Agent: {answer}", flush=True)
            if spoken_reply:
                try:
                    speaker.speak(answer)
                except KeyboardInterrupt:
                    print("\nSpeech stopped.")
                except Exception as error:
                    print(f"Speech error: {error}")
    except (EOFError, KeyboardInterrupt):
        print()
