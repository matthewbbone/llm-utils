from dotenv import load_dotenv
import os
import openai
from openai import (
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    APIError,
    APIConnectionError,
    APITimeoutError
)
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

        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("OPENAI_KEY environment variable not set")
        self.client = openai.Client(api_key=api_key)
        
    def _get_default_batch_size(self, num_texts):
        # openai limits to 2048 per request
        batch_size = min(2048 * os.cpu_count(), num_texts)
        # chroma limits to 5461 uploads
        batch_size = min(5461, batch_size)
        return batch_size
        
    def _embedding_call(self, texts, embedding_model, size, max_retries=5):
        attempt = 0
        delay = 1
        last_error = None

        while attempt < max_retries:
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=embedding_model,
                    dimensions=size
                )
                return [e.embedding for e in response.data]

            except AuthenticationError as e:
                print(f"OpenAI authentication failed: {e}")
                return None

            except BadRequestError as e:
                print(f"OpenAI bad request (check model/input): {e}")
                return None

            except RateLimitError as e:
                last_error = e
                attempt += 1
                wait_time = delay * 2  # Longer wait for rate limits
                print(f"OpenAI rate limit hit, waiting {wait_time}s (attempt {attempt}/{max_retries})")
                time.sleep(wait_time)
                delay *= 2

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                attempt += 1
                print(f"OpenAI connection error, retrying (attempt {attempt}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

            except APIError as e:
                last_error = e
                attempt += 1
                print(f"OpenAI API error, retrying (attempt {attempt}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

            except Exception as e:
                last_error = e
                attempt += 1
                print(f"Unexpected error in embedding call (attempt {attempt}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

        print(f"OpenAI embedding failed after {max_retries} attempts. Last error: {last_error}")
        return None 
    
    def save_embeddings(
        self, 
        ids,
        texts, 
        size, 
        db,
        name,
        verbose=True,
        model=None
    ):
    
        output_texts, output_embeddings = self._concurrent_embedding_call(
            ids,
            texts, 
            model, 
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
        last_error = None

        if web_search:
            tools = [{"type": "web_search"}]
        else:
            tools = []

        # Map unified reasoning levels to OpenAI format
        # OpenAI expects: reasoning={"effort": "low"|"medium"|"high"}
        # "minimal" maps to "low" since OpenAI doesn't have "minimal"
        reasoning_param = None
        if reasoning:
            effort_mapping = {
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high"
            }
            if reasoning in effort_mapping:
                reasoning_param = {"effort": effort_mapping[reasoning]}

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
                    reasoning=reasoning_param
                )

                if (isinstance(response_format, str) and "json" in response_format) or (isinstance(response_format, dict) and response_format.get("type") == "json_schema"):
                    return json.loads(response.output_text)
                else:
                    return response.output_text

            except AuthenticationError as e:
                return {"error": f"Authentication failed: {e}", "error_type": "auth"}

            except BadRequestError as e:
                return {"error": f"Bad request (check model/parameters): {e}", "error_type": "bad_request"}

            except RateLimitError as e:
                last_error = e
                attempt += 1
                wait_time = delay * 2
                if attempt < max_retries:
                    print(f"OpenAI rate limit hit, waiting {wait_time}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait_time)
                    delay *= 2

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"OpenAI connection error, retrying (attempt {attempt}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2

            except APIError as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"OpenAI API error, retrying (attempt {attempt}/{max_retries}): {e}")
                    time.sleep(delay)
                    delay *= 2

            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse JSON response: {e}", "error_type": "parse_error"}

            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"Unexpected error, retrying (attempt {attempt}/{max_retries}): {e}")
                    time.sleep(delay)
                    delay *= 2

        return {"error": f"Failed after {max_retries} attempts: {last_error}", "error_type": "max_retries"}

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
    
    
