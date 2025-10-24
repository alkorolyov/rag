from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class BaseLLM(ABC):
    @abstractmethod
    def generate(
            self,
            prompt: str,
            system_prompt: str,
            max_tokens: int,
            temperature: float,
            **kwargs,
    ) -> str:
        """Generate text from a given prompt."""
        ...

