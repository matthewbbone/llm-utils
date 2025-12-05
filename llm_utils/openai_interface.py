from dotenv import load_dotenv
import os
import openai
import time
from datetime import datetime as dt
import numpy as np
import concurrent.futures
from tqdm import tqdm
import ast
import json
import pandas as pd
from llm_utils.llm_interface import LLMInterface

class OpenAIInterface(LLMInterface):

    def __init__(self) -> None:
        super().__init__()
        load_dotenv()
        
        self.client = openai.Client(api_key=os.getenv("OPENAI_KEY"))
        
    def _get_default_batch_size(self, num_texts):
        # openai limits to 2048 per request
        batch_size = min(2048 * os.cpu_count(), num_texts)
        # chroma limits to 5461 uploads
        batch_size = min(5461, batch_size)
        return batch_size
        
    def _embedding_call(self, texts, embedding_model, size, max_retries=5):
        
        attempt = 0
        delay = 1
        while attempt < max_retries:
            
            try:
                response = self.client.embeddings.create(
                    input=texts, 
                    model=embedding_model,
                    dimensions=size
                )
                embeddings = [e.embedding for e in response.data]  
                return embeddings
            except Exception as e:
                attempt += 1
                time.sleep(delay)
                delay *= 2  # Exponential backoff
           
        print(f"Problematic text: {texts}")
        return None 
    
    def save_embeddings(
        self, 
        ids,
        texts, 
        size, 
        db,
        name,
        batch=False,
        verbose=True,
        model=None
    ):
        
        if model:
            embedding_model = model
        elif size > 1536:
            embedding_model = "text-embedding-3-large"
        else:
            embedding_model = "text-embedding-3-large"
    
        output_texts, output_embeddings = self._concurrent_embedding_call(
            ids,
            texts, 
            embedding_model, 
            size,
            db,
            verbose=verbose
        )
        return output_texts, output_embeddings


    def _chat_completion_call(
        self,
        model,
        message,
        web_search,
        max_tokens,
        max_tool_calls,
        tool_choice,
        system_message,
        response_format,
        verbosity,
        reasoning
    ):
        max_retries = 5
        attempt = 0
        delay = 1
        
        if web_search:
            tools = [{ "type": "web_search" }]
        else:
            tools = []
        
        while attempt < max_retries:
            try:
                response = self.client.responses.create(
                    model=model,
                    input=message,
                    tools=tools,
                    max_output_tokens=max_tokens,
                    max_tool_calls=max_tool_calls,
                    tool_choice=tool_choice,
                    instructions=system_message,
                    text={"format": response_format, "verbosity": verbosity},
                    reasoning=reasoning
                )

                if (isinstance(response_format, str) and "json" in response_format) or (isinstance(response_format, dict) and response_format.get("type") == "json_schema"):
                    return json.loads(response.output_text)
                else:
                    return response.output_text
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"Error in _chat_completion_call after {max_retries} attempts: {e}")
                    return {"error": str(e)}

    def ask_gpt(
        self,
        system_message,
        ids,
        user_messages,
        response_formats,
        model,
        web_search=False,
        max_tool_calls=3,
        tool_choice=None,
        reasoning=None,
        verbosity="medium",
        max_tokens=None,
        n_workers=1
    ):
    
        return self._chat_completion(
            system_message,
            ids,
            user_messages,
            response_formats,
            model,
            web_search=web_search,
            max_tool_calls=max_tool_calls,
            tool_choice=tool_choice,
            reasoning=reasoning,
            verbosity=verbosity,
            max_tokens=max_tokens,
            n_workers=n_workers
        )
    
    
