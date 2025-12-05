from llm_utils.openai_interface import OpenAIInterface
from llm_utils.gemini_interface import GeminiInterface
from llm_utils.claude_interface import ClaudeInterface

class LLMWrapper:
    def __init__(self):
        self.openai = OpenAIInterface()
        self.gemini = GeminiInterface()
        self.claude = ClaudeInterface()

    def ask(
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
        n_workers=1,
    ):
        if "gpt" in model:
            return self.openai.ask(
                system_message,
                ids,
                user_messages,
                response_formats,
                model,
                web_search,
                max_tool_calls,
                tool_choice,
                reasoning,
                verbosity,
                max_tokens,
                n_workers
            )
        elif "gemini" in model:
            return self.gemini.ask(
                system_message,
                ids,
                user_messages,
                response_formats,
                model,
                web_search,
                max_tool_calls,
                tool_choice,
                reasoning,
                verbosity,
                max_tokens,
                n_workers
            )
        elif "claude" in model:
            return self.claude.ask(
                system_message,
                ids,
                user_messages,
                response_formats,
                model,
                web_search,
                max_tool_calls,
                tool_choice,
                reasoning,
                verbosity,
                max_tokens,
                n_workers
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

    def embed(self, model=None, **kwargs):
        if model:
            kwargs["model"] = model
            if model in ["gemini-embedding-001"]:
                return self.gemini.save_embeddings(**kwargs)
            elif model in ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002	"]:
                return self.openai.save_embeddings(**kwargs)
            else:
                # Default fallback or error if model format is unknown
                # Assuming openai for generic text-embedding if not 004
                return self.openai.save_embeddings(**kwargs)
        else:
            # Fallback to provider arg if model is not provided, for backward compatibility
            provider = kwargs.pop("provider", "openai")
            if provider == "openai":
                return self.openai.save_embeddings(**kwargs)
            elif provider == "gemini":
                return self.gemini.save_embeddings(**kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
