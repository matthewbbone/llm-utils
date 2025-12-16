import os
from google.oauth2 import service_account
from google.genai import Client
from google.api_core import exceptions as google_exceptions
import time
from google import genai
from llm_utils.llm_interface import LLMInterface


class GeminiInterface(LLMInterface):

    def __init__(self) -> None:
        super().__init__()

        # Check for API key (GenAI client uses GOOGLE_API_KEY or GEMINI_API_KEY by default)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY or GEMINI_API_KEY not set. Using default credentials.")

        self.client = Client()
        
    def _embedding_call(self, texts, embedding_model, size, max_retries=5):
        attempt = 0
        delay = 1
        last_error = None

        while attempt < max_retries:
            try:
                response = self.client.models.embed_content(
                    model=embedding_model,
                    contents=texts,
                    config=genai.types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=size
                    )
                )
                return [e.values for e in response.embeddings]

            except google_exceptions.Unauthenticated as e:
                print(f"Gemini authentication failed: {e}")
                return None

            except google_exceptions.InvalidArgument as e:
                print(f"Gemini invalid argument (check model/input): {e}")
                return None

            except google_exceptions.ResourceExhausted as e:
                last_error = e
                attempt += 1
                wait_time = delay * 2
                print(f"Gemini rate limit hit, waiting {wait_time}s (attempt {attempt}/{max_retries})")
                time.sleep(wait_time)
                delay *= 2

            except (google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded) as e:
                last_error = e
                attempt += 1
                print(f"Gemini service error, retrying (attempt {attempt}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

            except google_exceptions.GoogleAPIError as e:
                last_error = e
                attempt += 1
                print(f"Gemini API error, retrying (attempt {attempt}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

            except Exception as e:
                last_error = e
                attempt += 1
                print(f"Unexpected error in embedding call (attempt {attempt}/{max_retries}): {e}")
                time.sleep(delay)
                delay *= 2

        print(f"Gemini embedding failed after {max_retries} attempts. Last error: {last_error}")
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
        
        # Map tools
        gemini_tools = []

        if web_search:
            gemini_tools.append(genai.types.Tool(google_search=genai.types.GoogleSearch()))
        
        # Extract schema from OpenAI-style format if present
        if isinstance(response_format, dict) and "schema" in response_format and "type" in response_format and response_format["type"] == "json_schema":
            response_format = response_format["schema"]

        def clean_schema(schema):
            if isinstance(schema, dict):
                return {k: clean_schema(v) for k, v in schema.items() if k not in ["max_completion_tokens", "additionalProperties"]}
            elif isinstance(schema, list):
                return [clean_schema(v) for v in schema]
            else:
                return schema
        
        response_format = clean_schema(response_format)

        last_error = None

        while attempt < max_retries:
            try:
                if web_search:
                    # Step 1: Search
                    search_config = genai.types.GenerateContentConfig(
                        tools=gemini_tools,
                        system_instruction=system_message
                    )

                    search_response = self.client.models.generate_content(
                        model=model,
                        contents=message,
                        config=search_config
                    )

                    search_text = search_response.text

                    # Step 2: Format
                    format_config = genai.types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        response_schema=response_format,
                        system_instruction=system_message
                    )

                    format_prompt = f"Original Request: {message}\n\nInformation Found: {search_text}\n\nPlease format the information found according to the schema."

                    response = self.client.models.generate_content(
                        model=model,
                        contents=format_prompt,
                        config=format_config
                    )

                    return response.parsed

                else:
                    # Standard single call
                    config = genai.types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        response_schema=response_format,
                        system_instruction=system_message
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=message,
                        config=config
                    )

                    return response.parsed

            except google_exceptions.Unauthenticated as e:
                return {"error": f"Authentication failed: {e}", "error_type": "auth"}

            except google_exceptions.InvalidArgument as e:
                return {"error": f"Invalid argument (check model/parameters): {e}", "error_type": "bad_request"}

            except google_exceptions.ResourceExhausted as e:
                last_error = e
                attempt += 1
                wait_time = delay * 2
                if attempt < max_retries:
                    print(f"Gemini rate limit hit, waiting {wait_time}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait_time)
                    delay *= 2

            except (google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded) as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"Gemini service error, retrying (attempt {attempt}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2

            except google_exceptions.GoogleAPIError as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"Gemini API error, retrying (attempt {attempt}/{max_retries}): {e}")
                    time.sleep(delay)
                    delay *= 2

            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"Unexpected error, retrying (attempt {attempt}/{max_retries}): {e}")
                    time.sleep(delay)
                    delay *= 2

        return {"error": f"Failed after {max_retries} attempts: {last_error}", "error_type": "max_retries"}
                
    def ask_gemini(
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