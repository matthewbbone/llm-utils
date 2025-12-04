from llm_utils.openai_interface import OpenAIInterface
from llm_utils.gemini_interface import GeminiInterface
import os
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class MathAnswer(BaseModel):
    answer: float
    explanation: str

class CapitalAnswer(BaseModel):
    capital: str
    country: str
    
class StockAnswer(BaseModel):
    symbol: str
    price: float
    currency: str

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def test_embeddings(openai_llm, gemini_llm):
    print("\n--- Testing Embeddings ---")
    
    texts = ["cat", "kitty", "dog", "potato"]
    ids = ["1", "2", "3", "4"]
    
    # OpenAI Embeddings
    print("Getting OpenAI Embeddings...")
    try:
        _, openai_embeddings = openai_llm.save_embeddings(
            ids=ids,
            texts=texts,
            size=1536,
            db=None,
            name="test_openai_emb",
            verbose=False
        )
        
        cat_emb = np.array(openai_embeddings[0])
        kitty_emb = np.array(openai_embeddings[1])
        dog_emb = np.array(openai_embeddings[2])
        potato_emb = np.array(openai_embeddings[3])
        
        sim_cat_kitty = cosine_similarity(cat_emb, kitty_emb)
        sim_dog_potato = cosine_similarity(dog_emb, potato_emb)
        
        print(f"OpenAI Similarity (cat, kitty): {sim_cat_kitty:.4f}")
        print(f"OpenAI Similarity (dog, potato): {sim_dog_potato:.4f}")
        
        if sim_cat_kitty > sim_dog_potato:
            print("✅ OpenAI Embedding Test Passed: cat-kitty similarity > dog-potato similarity")
        else:
            print("❌ OpenAI Embedding Test Failed")
            
    except Exception as e:
        print("OpenAI Embedding Test Failed:", e)

    # Gemini Embeddings
    print("Getting Gemini Embeddings...")
    try:
        _, gemini_embeddings = gemini_llm.save_embeddings(
            ids=ids,
            texts=texts,
            size=768, 
            db=None,
            name="test_gemini_emb",
            verbose=False
        )
        
        cat_emb = np.array(gemini_embeddings[0])
        kitty_emb = np.array(gemini_embeddings[1])
        dog_emb = np.array(gemini_embeddings[2])
        potato_emb = np.array(gemini_embeddings[3])
        
        sim_cat_kitty = cosine_similarity(cat_emb, kitty_emb)
        sim_dog_potato = cosine_similarity(dog_emb, potato_emb)
        
        print(f"Gemini Similarity (cat, kitty): {sim_cat_kitty:.4f}")
        print(f"Gemini Similarity (dog, potato): {sim_dog_potato:.4f}")
        
        if sim_cat_kitty > sim_dog_potato:
            print("✅ Gemini Embedding Test Passed: cat-kitty similarity > dog-potato similarity")
        else:
            print("❌ Gemini Embedding Test Failed")

    except Exception as e:
        print("Gemini Embedding Test Failed:", e)

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
    
    system_message = "You are a helpful assistant."
    user_messages = ["What is 2+2?", "What is the capital of France?", "What is the current stock price of TSLA?"]
    ids = ["1", "2", "3"]
    # Use Pydantic models for response format
    response_formats = [MathAnswer, CapitalAnswer, StockAnswer]
    
    print("\n--- Testing OpenAI Chat (Structured) ---")
    try:
        openai_results = openai_llm.ask_gpt(
            system_message=system_message,
            ids=ids,
            user_messages=user_messages,
            response_formats=response_formats,
            model="gpt-4o-mini", 
            name="test_openai",
            web_search=True
        )
        print("OpenAI Results:", openai_results)
        # Verify types
        if isinstance(openai_results['1'], MathAnswer) and isinstance(openai_results['2'], CapitalAnswer) and isinstance(openai_results['3'], StockAnswer):
             print("✅ OpenAI Structured Output Test Passed")
        else:
             print("❌ OpenAI Structured Output Test Failed (Types mismatch)")
             print(f"Type 1: {type(openai_results.get('1'))}")
             print(f"Type 2: {type(openai_results.get('2'))}")
             print(f"Type 3: {type(openai_results.get('3'))}")

    except Exception as e:
        print("OpenAI Test Failed:", e)
        import traceback
        traceback.print_exc()

    print("\n--- Testing Gemini Chat (Structured) ---")
    try:
        gemini_results = gemini_llm.ask_gemini(
            system_message=system_message,
            ids=ids,
            user_messages=user_messages,
            response_formats=response_formats,
            model="gemini-2.5-flash",
            name="test_gemini",
            web_search=True
        )
        print("Gemini Results:", gemini_results)
        # Verify types
        if isinstance(gemini_results['1'], MathAnswer) and isinstance(gemini_results['2'], CapitalAnswer) and isinstance(gemini_results['3'], StockAnswer):
             print("✅ Gemini Structured Output Test Passed")
        else:
             print("❌ Gemini Structured Output Test Failed (Types mismatch)")
             print(f"Type 1: {type(gemini_results.get('1'))}")
             print(f"Type 2: {type(gemini_results.get('2'))}")
             print(f"Type 3: {type(gemini_results.get('3'))}")

    except Exception as e:
        print("Gemini Test Failed:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_interfaces()
