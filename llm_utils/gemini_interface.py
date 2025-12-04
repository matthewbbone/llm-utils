from .llm_interface import LLMInterface
from dotenv import load_dotenv
import os
from google import genai
import time
import ast
from google.oauth2 import service_account
from google.genai import Client
from pydantic import BaseModel

class GeminiInterface(LLMInterface):

    def __init__(self) -> None:
        super().__init__()
        
        credentials= {
            "type": os.environ["GEMINI_TYPE"],
            "project_id": os.environ["GEMINI_PROJECT_ID"],
            "private_key_id": os.environ["GEMINI_PRIVATE_KEY_ID"],
            "private_key": os.environ["GEMINI_PRIVATE_KEY"].replace("\\n", "\n"),
            "client_email": os.environ["GEMINI_CLIENT_EMAIL"],
            "client_id": os.environ["GEMINI_CLIENT_ID"],
            "auth_uri": os.environ["GEMINI_AUTH_URI"],
            "token_uri": os.environ["GEMINI_TOKEN_URI"],
            "auth_provider_x509_cert_url": os.environ["GEMINI_AUTH_PROVIDER_CERT"],
            "client_x509_cert_url": os.environ["GEMINI_CLIENT_CERT_URL"],
            "universe_domain": os.environ["GEMINI_UNIVERSE_DOMAIN"]
        }
        
        creds = service_account.Credentials.from_service_account_info(credentials)
        creds = creds.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
        
        self.client = Client(
            vertexai=True,
            project=credentials["project_id"],
            location="us-central1", # You must specify the location (e.g., us-central1, us-east4)
            credentials=creds
        )
        
    def _embedding_call(self, texts, embedding_model, size, max_retries=5):
        
        attempt = 0
        delay = 1
        while attempt < max_retries:
            
            try:
                # Gemini embedding call
                response = self.client.models.embed_content(
                    model=embedding_model,
                    contents=texts,
                    config=genai.types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=size
                    )
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                attempt += 1
                time.sleep(delay)
                delay *= 2  # Exponential backoff
           
        print(f"Problematic text: {texts}")
        return None 
    
    def _get_default_batch_size(self, num_texts):
        return 100

    def save_embeddings(
        self, 
        ids,
        texts, 
        size, 
        db,
        name,
        verbose=True
    ):
        
        embedding_model = "text-embedding-004"

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
        
        # Map tools
        gemini_tools = []
        if tools:
            for tool in tools:
                if tool.get("type") == "web_search_preview":
                    # Enable Google Search grounding
                    gemini_tools.append(genai.types.Tool(google_search=genai.types.GoogleSearch()))
        
        # Generation config
        mime_type = "text/plain"
        response_schema = None

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            mime_type = "application/json"
            response_schema = response_format
        elif isinstance(response_format, dict) and response_format.get("type") == "json_object":
            mime_type = "application/json"
        elif response_format == "json_object":
            mime_type = "application/json"

        config = genai.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            response_mime_type=mime_type,
            response_schema=response_schema,
            tools=gemini_tools if gemini_tools else None,
            system_instruction=system_message
        )

        while attempt < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=message,
                    config=config
                )
                
                if response_schema:
                    return response.parsed

                text = response.text.strip()
                # Remove markdown code blocks if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    if text.endswith("```"):
                        text = text[:-3]
                
                return ast.literal_eval(text.strip())
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"Error in _chat_completion_call after {max_retries} attempts: {e}")
                    return {"error": str(e)}

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
