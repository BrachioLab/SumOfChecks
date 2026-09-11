"""OpenAI GPT model adapter."""

import concurrent.futures
import os
import time
from typing import Any, List, Optional, Sequence, Union

from openai import OpenAI

from .base import Batch, OneQuery
from .utils import cache, get_cache_key, image_to_base64, is_image


class OpenAIAdapter:
    """OpenAI API adapter for GPT models."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        max_tokens: int = 2048,
        batch_size: int = 24,
        use_cache: bool = True,
        verbose: bool = True,
        temperature: float = 0.1,
    ):
        """Initialize OpenAI adapter.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (uses env var if not provided)
            num_tries_per_request: Number of retries for failed requests
            max_tokens: Maximum tokens in response
            batch_size: Maximum concurrent requests
            use_cache: Whether to use disk cache
            verbose: Whether to print error messages
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.temperature = temperature

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=self.api_key)

    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        """Process a batch of prompts through the model.

        Args:
            prompts: Batch of queries
            system_prompt: System prompt

        Returns:
            List of model responses
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [
                executor.submit(self._one_call, prompt=p, system_prompt=system_prompt)
                for p in prompts
            ]
            return [f.result() for f in futures]

    def _one_call(self, prompt: OneQuery, system_prompt: str) -> str:
        """Process a single prompt.

        Args:
            prompt: Single query tuple
            system_prompt: System prompt

        Returns:
            Model response string
        """
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt, system_prompt))
            if ret is not None and ret != "":
                return ret

        # Build user content from tuple
        user_content = []
        for p in prompt:
            if isinstance(p, str):
                user_content.append({"type": "text", "text": p})
            elif is_image(p):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_to_base64(p, 'PNG')}"}
                })
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response_text = ""
        last_error: Optional[Exception] = None
        for _ in range(self.num_tries_per_request):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                response_text = (response.choices[0].message.content or "").strip()
                if response_text:
                    break
            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"❌ Error calling OpenAI API for {self.model_name}: {e}")
                    print(f"   Attempt {_ + 1}/{self.num_tries_per_request}")
                time.sleep(3)

        if not response_text:
            error_msg = (
                f"Empty response from OpenAI model '{self.model_name}' "
                f"after {self.num_tries_per_request} attempts."
            )
            if last_error is not None:
                error_msg += f" Last error: {type(last_error).__name__}: {last_error}"
            raise RuntimeError(error_msg)

        if self.use_cache and response_text:
            cache.set(get_cache_key(self.model_name, prompt, system_prompt), response_text)

        return response_text
