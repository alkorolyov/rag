from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class BaseLLM(ABC):
    @abstractmethod
    def generate(
            self,
            prompt: str,
            max_tokens: int = 512,
            temperature: float = 1.0,
            **kwargs,
    ) -> str:
        """Generate text from a given prompt."""
        ...

