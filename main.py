# main.py
import os
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent_config.agent import root_agent

# Load your local variables
load_dotenv()

def main():
    if not os.getenv("GOOGLE_API_KEY") or "YourGeminiAPIKey" in os.getenv("GOOGLE_API_KEY"):
        print("⚠️ Warning: Please put your real Gemini API key in the '.env' file!")
        return

    # Map GOOGLE_API_KEY to GEMINI_API_KEY as expected by the Google SDK
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    # Set up memory sessions
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="pisu_app", session_service=session_service)

    print("🤖 PISU Agent successfully initialized! Type 'exit' to quit.")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        new_message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )

        try:
            events = runner.run(
                user_id="developer_1",
                session_id="local_dev_session",
                new_message=new_message
            )
            for event in events:
                if event.is_final_response():
                    print(f"\nAgent: {event.content.parts[0].text}")
        except Exception as e:
            print(f"\nAn error occurred running the agent: {e}")

if __name__ == "__main__":
    main()