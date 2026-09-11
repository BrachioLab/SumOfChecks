"""Provider adapters used by the Sum-of-Checks paper."""

from .base import Batch, ModelAdapter, OneQuery, PromptPart


def create_model(model_id, use_cache=True, verbose=True, dataset=None, temperature=0.1):
    if model_id == "gpt-4.1-mini":
        from .openai_gpt import OpenAIAdapter
        adapter_class = OpenAIAdapter
    elif model_id in ("claude-haiku-4-5-20251001", "claude-opus-4-5-20251101"):
        from .anthropic_claude import AnthropicAdapter
        adapter_class = AnthropicAdapter
    else:
        raise ValueError(f"Unsupported paper model: {model_id}")
    adapter = adapter_class(model_name=model_id, use_cache=use_cache,
                            verbose=verbose, temperature=temperature)
    adapter.model_id = model_id
    return adapter
