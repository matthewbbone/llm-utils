import os
from dotenv import load_dotenv
import anthropic
from anthropic import (
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    APIError,
    APIConnectionError,
    APITimeoutError
)
import time
import json
from llm_utils.llm_interface import LLMInterface

class ClaudeInterface(LLMInterface):

    def __init__(self) -> None:
        super().__init__()
        load_dotenv()

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)

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

        last_error = None

        # Determine if model is Opus 4.5 (supports effort parameter)
        is_opus_45 = "claude-opus-4-5" in model or "opus-4-5" in model

        # Map unified reasoning levels to Claude
        # For Opus 4.5: use effort parameter (beta API)
        # For other models: use extended thinking with budget_tokens
        effort_mapping = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high"
        }

        # Token budgets for extended thinking (non-Opus models)
        thinking_budget_mapping = {
            "minimal": 1024,
            "low": 4096,
            "medium": 16384,
            "high": 65536
        }

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

                # Add reasoning/thinking based on model type
                if reasoning and reasoning in effort_mapping:
                    if is_opus_45:
                        # Opus 4.5: use effort parameter (beta API)
                        kwargs["betas"] = ["effort-2025-11-24"]
                        kwargs["output_config"] = {"effort": effort_mapping[reasoning]}
                        response = self.client.beta.messages.create(**kwargs)
                    else:
                        # Other models: use extended thinking with budget_tokens
                        budget = thinking_budget_mapping[reasoning]
                        kwargs["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": budget
                        }
                        # Ensure max_tokens can accommodate thinking + response
                        if kwargs["max_tokens"] < budget + 1024:
                            kwargs["max_tokens"] = budget + 4096
                        response = self.client.messages.create(**kwargs)
                else:
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

            except AuthenticationError as e:
                return {"error": f"Authentication failed: {e}", "error_type": "auth"}

            except BadRequestError as e:
                return {"error": f"Bad request (check model/parameters): {e}", "error_type": "bad_request"}

            except RateLimitError as e:
                last_error = e
                attempt += 1
                wait_time = delay * 2
                if attempt < max_retries:
                    print(f"Claude rate limit hit, waiting {wait_time}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait_time)
                    delay *= 2

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"Claude connection error, retrying (attempt {attempt}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2

            except APIError as e:
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    print(f"Claude API error, retrying (attempt {attempt}/{max_retries}): {e}")
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