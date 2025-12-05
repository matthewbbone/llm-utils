import os
from dotenv import load_dotenv
import anthropic
import time
import json
from llm_utils.llm_interface import LLMInterface

class ClaudeInterface(LLMInterface):

    def __init__(self) -> None:
        super().__init__()
        load_dotenv()
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _embedding_call(self, texts, embedding_model, size, max_retries=5):
        # Anthropic does not support embeddings natively
        return None

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
        claude_tools = []
        tool_choice_param = None
        
        if web_search:
            claude_tools.append({
                "type": "web_search_20250305",
                "name": "web_search"
            })
        
        # Handle structured output via tools
        if response_format:
            # If web search is enabled, we can't force tool choice immediately if we want search to happen first.
            # However, Claude's web search is a tool itself.
            # If we force a specific tool, Claude might skip web search.
            # For now, if web search is enabled, we don't force tool choice, but provide the schema tool.
            # We rely on the prompt to encourage using the schema tool after searching.
            
            schema = None
            name = "output_schema"
            description = ""
            
            if isinstance(response_format, dict):
                if "schema" in response_format:
                    schema = response_format["schema"]
                    name = response_format.get("name", name)
                    description = response_format.get("description", description)
                elif "json_schema" in response_format:
                    # OpenAI format
                    json_schema = response_format["json_schema"]
                    schema = json_schema.get("schema")
                    name = json_schema.get("name", name)
                    description = json_schema.get("description", description)
            
            if schema:
                # Clean schema for Claude
                def clean_schema(s):
                    if isinstance(s, dict):
                        return {k: clean_schema(v) for k, v in s.items() if k not in ["max_completion_tokens", "additionalProperties"]}
                    elif isinstance(s, list):
                        return [clean_schema(v) for v in s]
                    else:
                        return s
                
                schema = clean_schema(schema)
                
                tool = {
                    "name": name,
                    "description": description,
                    "input_schema": schema
                }
                claude_tools.append(tool)
                
                if not web_search:
                    tool_choice_param = {"type": "tool", "name": name}
                else:
                    # If web search is on, we let Claude decide when to use the output tool
                    # But we should update the system message to ensure it uses the tool eventually
                    system_message += f"\n\nAfter gathering necessary information, you MUST call the '{name}' tool to provide your final answer."

        while attempt < max_retries:
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": max_tokens if max_tokens else 4096,
                    "system": system_message,
                    "timeout": 600
                }
                
                if claude_tools:
                    kwargs["tools"] = claude_tools
                    if tool_choice_param:
                        kwargs["tool_choice"] = tool_choice_param
                
                response = self.client.messages.create(**kwargs)
                
                # Parse response
                # If structured output was requested (response_format is set), look for that tool use
                if response_format:
                    target_tool_name = "output_schema"
                    if isinstance(response_format, dict):
                        if "name" in response_format:
                            target_tool_name = response_format["name"]
                        elif "json_schema" in response_format and "name" in response_format["json_schema"]:
                            target_tool_name = response_format["json_schema"]["name"]

                    for content in response.content:
                        if content.type == "tool_use" and content.name == target_tool_name:
                            return content.input
                
                # If no structured output forced, or if we just want text (e.g. web search results integrated)
                # Claude returns text blocks with citations for web search
                text_content = []
                for content in response.content:
                    if content.type == "text":
                        text_content.append(content.text)
                
                if text_content:
                    return "".join(text_content)
                
                return None

            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"Error in _chat_completion_call after {max_retries} attempts: {e}")
                    return {"error": str(e)}