"""Base model adapter protocol."""

from typing import Sequence, Tuple, Union
from abc import ABC, abstractmethod

from PIL import Image

# Type aliases for prompts
PromptPart = Union[str, Image.Image]
OneQuery = Tuple[PromptPart, ...]
Batch = Sequence[OneQuery]


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    All model implementations must follow this interface to ensure
    compatibility with the endopoint system.
    """

    @abstractmethod
    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        """Process a batch of prompts through the model.

        Args:
            prompts: Batch of queries, each a tuple of text/image parts
            system_prompt: System prompt to use for all queries

        Returns:
            List of model responses, one per query
        """
        pass
