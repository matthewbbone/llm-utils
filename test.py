from llm_utils.openai_interface import OpenAIInterface
from llm_utils.gemini_interface import GeminiInterface
import os
from dotenv import load_dotenv

load_dotenv()

def test_interfaces():
    print("Testing Interfaces...")
    
    # Ensure keys are present (mocking or checking)
    if not os.getenv("OPENAI_KEY"):
        print("Warning: OPENAI_KEY not found.")
    if not os.getenv("GEMINI_PRIVATE_KEY_ID"):
        print("Warning: GEMINI_PRIVATE_KEY_ID not found.")

    try:
        print("Initializing OpenAI Interface...")
        openai_llm = OpenAIInterface()
        print("Initializing Gemini Interface...")
        gemini_llm = GeminiInterface()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error initializing interfaces: {e}")
        return
    
    system_message = "You are a helpful assistant. Respond with a Python dictionary containing the key 'answer'."
    user_messages = ["What is 2+2?", "What is the capital of France?"]
    ids = ["1", "2"]
    response_formats = [None, None]
    
    print("\n--- Testing OpenAI Chat ---")
    try:
        openai_results = openai_llm.ask_gpt(
            system_message=system_message,
            ids=ids,
            user_messages=user_messages,
            response_formats=response_formats,
            model="gpt-4o-mini", 
            name="test_openai"
        )
        print("OpenAI Results:", openai_results)
    except Exception as e:
        print("OpenAI Test Failed:", e)

    print("\n--- Testing Gemini Chat ---")
    try:
        gemini_results = gemini_llm.ask_gemini(
            system_message=system_message,
            ids=ids,
            user_messages=user_messages,
            response_formats=response_formats,
            model="gemini-2.5-flash",
            name="test_gemini"
        )
        print("Gemini Results:", gemini_results)
    except Exception as e:
        print("Gemini Test Failed:", e)

if __name__ == "__main__":
    test_interfaces()
