from .llm_interface import LLMInterface
from dotenv import load_dotenv
import os
import openai
import time
import ast
from pydantic import BaseModel

class OpenAIInterface(LLMInterface):

    def __init__(self) -> None:
        super().__init__()
        self.client = openai.Client(api_key=os.getenv("OPENAI_KEY"))
        
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
    
    def _get_default_batch_size(self, num_texts):
        # openai limits to 2048 per request
        batch_size = min(2048 * os.cpu_count(), num_texts)
        # chroma limits to 5461 uploads
        return min(5461, batch_size)

    def save_embeddings(
        self, 
        ids,
        texts, 
        size, 
        db,
        name,
        verbose=True
    ):
        
        if size > 1536:
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
        tools,
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
        
        while attempt < max_retries:
            try:
                kwargs = {
                    "model": model,
                    "input": message,
                    "instructions": system_message,
                    "tools": tools if tools else None,
                    "tool_choice": tool_choice,
                    "max_output_tokens": max_tokens,
                }

                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_format.__name__,
                            "schema": response_format.model_json_schema(),
                            "strict": True
                        }
                    }
                elif response_format == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = self.client.responses.create(**kwargs)

                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    return response_format.model_validate_json(response.output_text)
                
                return ast.literal_eval(response.output_text)
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
        name,
        web_search=False,
        max_tool_calls=3,
        tool_choice=None,
        reasoning=None,
        verbosity="medium",
        max_tokens=1000,
        n_workers=1,
        from_cache=None
    ):
    
        return self.ask(
            system_message,
            ids,
            user_messages,
            response_formats,
            model,
            name,
            web_search,
            max_tool_calls,
            tool_choice,
            reasoning,
            verbosity,
            max_tokens,
            n_workers,
            from_cache
        )
    
    
