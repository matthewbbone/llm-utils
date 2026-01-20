from dotenv import load_dotenv
import os
from llm_utils.wrapper import LLMWrapper
import numpy as np
load_dotenv()

llm = LLMWrapper()

template = {
        "name": "test",
        "type": "json_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "COMPANY_NAME": {
                    "type": "string",
                    "description": "The name of the company.",
                },
                "BUSINESS_ACTIVITIES": {
                    "type": "string",
                    "description": "A description of the company's business activities.",
                },
                "PRICE": {
                    "type": "number",
                    "description": "The current stock price of the company in USD.",
                },
                "CITATIONS": {
                    "type": "array",
                    "description": "A list of urls for the information provided.",
                    "items": {
                        "type": "string",
                    }
                }
            },
            "required": ["COMPANY_NAME", "BUSINESS_ACTIVITIES", "PRICE", "CITATIONS"],
            "additionalProperties": False
        },
    }

print("Testing OpenAI...")
try:
    oai_res = llm.ask(
        "You are a financial analyst with expertise of ticker symbols and company information. Search the web for information about the company with ticker symbol TSLA.",
        [0],
        ["What company does this refer to? Describe it's business activities and price: TSLA"],
        [template],
        "gpt-5-mini",
        web_search=True,
        verbosity="medium",
        reasoning="low"
    )

    if 0 in oai_res and isinstance(oai_res[0], dict):
        price = oai_res[0].get("PRICE")
        if price and 400 < price < 500:
            print("✅ OpenAI Price Check Passed")
        else:
            print(f"❌ OpenAI Price Check Failed: {price}")
        print("Rest of answer:", {k:v for k,v in oai_res[0].items() if k != "PRICE"})
    else:
        print("❌ OpenAI Result format incorrect or error:", oai_res)
except Exception as e:
    print(f"❌ OpenAI Test Error: {e}")


print("\nTesting Gemini...")
try:
    gemini_res = llm.ask(
        "You are a financial analyst with expertise of ticker symbols and company information. Search the web for information about the company with ticker symbol TSLA.",
        [0],
        ["What company does this refer to? Describe it's business activities and price: TSLA"],
        [template],
        "gemini-3-pro-preview",
        web_search=True,
        reasoning="low",
        verbosity="medium"
    )

    if 0 in gemini_res and isinstance(gemini_res[0], dict):
        price = gemini_res[0].get("PRICE")
        print("Price:", price)
        if price and 400 < price < 500:
            print("✅ Gemini Price Check Passed")
        else:
            print(f"❌ Gemini Price Check Failed: {price}")
        print("Rest of answer:", {k:v for k,v in gemini_res[0].items() if k != "PRICE"})
    else:
        print("❌ Gemini Result format incorrect or error:", gemini_res)
except Exception as e:
    print(f"❌ Gemini Test Error: {e}")


print("\nTesting Claude...")
try:
    claude_res = llm.ask(
        "You are a financial analyst with expertise of ticker symbols and company information. Search the web for information about the company with ticker symbol TSLA.",
        [0],
        ["What company does this refer to? Describe it's business activities and price: TSLA"],
        [template],
        "claude-sonnet-4-5",
        web_search=True,
        verbosity="medium",
        reasoning="low"
    )

    if 0 in claude_res and isinstance(claude_res[0], dict):
        price = claude_res[0].get("PRICE")
        print("Price:", price)
        if price and 400 < price < 500:
            print("✅ Claude Price Check Passed")
        else:
            print(f"❌ Claude Price Check Failed: {price}")
        print("Rest of answer:", {k:v for k,v in claude_res[0].items() if k != "PRICE"})
    else:
        print("❌ Claude Result format incorrect or error:", claude_res)
except Exception as e:
    print(f"❌ Claude Test Error: {e}")


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# print("\n--- Testing Embeddings ---")
# try:
#     texts = ["cat", "kitty", "dog", "potato"]
#     ids = ["1", "2", "3", "4"]
    
#     # OpenAI Embeddings
#     print("Getting OpenAI Embeddings...")
#     _, openai_embeddings = llm.embed(
#         ids=ids,
#         texts=texts,
#         size=1536,
#         db=None,
#         name="test_openai_emb",
#         verbose=False,
#         model="text-embedding-3-large"
#     )
    
#     cat_emb = np.array(openai_embeddings[0])
#     kitty_emb = np.array(openai_embeddings[1])
#     dog_emb = np.array(openai_embeddings[2])
#     potato_emb = np.array(openai_embeddings[3])
    
#     sim_cat_kitty = cosine_similarity(cat_emb, kitty_emb)
#     sim_dog_potato = cosine_similarity(dog_emb, potato_emb)
    
#     print(f"OpenAI Similarity (cat, kitty): {sim_cat_kitty:.4f}")
#     print(f"OpenAI Similarity (dog, potato): {sim_dog_potato:.4f}")
    
#     if sim_cat_kitty > sim_dog_potato:
#         print("✅ OpenAI Embedding Test Passed: cat-kitty similarity > dog-potato similarity")
#     else:
#         print("❌ OpenAI Embedding Test Failed")

#     # Gemini Embeddings
#     print("Getting Gemini Embeddings...")
#     _, gemini_embeddings = llm.embed(
#         ids=ids,
#         texts=texts,
#         size=768, 
#         db=None,
#         name="test_gemini_emb",
#         verbose=False,
#         model="text-embedding-004"
#     )
    
#     cat_emb = np.array(gemini_embeddings[0])
#     kitty_emb = np.array(gemini_embeddings[1])
#     dog_emb = np.array(gemini_embeddings[2])
#     potato_emb = np.array(gemini_embeddings[3])
    
#     sim_cat_kitty = cosine_similarity(cat_emb, kitty_emb)
#     sim_dog_potato = cosine_similarity(dog_emb, potato_emb)
    
#     print(f"Gemini Similarity (cat, kitty): {sim_cat_kitty:.4f}")
#     print(f"Gemini Similarity (dog, potato): {sim_dog_potato:.4f}")
    
#     if sim_cat_kitty > sim_dog_potato:
#         print("✅ Gemini Embedding Test Passed: cat-kitty similarity > dog-potato similarity")
#     else:
#         print("❌ Gemini Embedding Test Failed")

# except Exception as e:
#     print(f"❌ Embedding Test Error: {e}")