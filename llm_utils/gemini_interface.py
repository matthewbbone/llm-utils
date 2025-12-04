import os
from google.oauth2 import service_account
from google.genai import Client
import time 
from google import genai
from llm_utils.llm_interface import LLMInterface


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
            location="us-east4", 
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
        
        # Check for web search in tools
        web_search = False
        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "web_search_preview":
                    web_search = True
                    break
        
        if web_search:
            gemini_tools.append(genai.types.Tool(google_search=genai.types.GoogleSearch()))
        
        # Extract schema from OpenAI-style format if present
        if isinstance(response_format, dict) and "schema" in response_format and "type" in response_format and response_format["type"] == "json_schema":
            response_format = response_format["schema"]

        def clean_schema(schema):
            if isinstance(schema, dict):
                return {k: clean_schema(v) for k, v in schema.items() if k != "max_completion_tokens"}
            elif isinstance(schema, list):
                return [clean_schema(v) for v in schema]
            else:
                return schema
        
        response_format = clean_schema(response_format)

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