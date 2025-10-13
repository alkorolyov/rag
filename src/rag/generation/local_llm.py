import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.generation.base import BaseLLM


class LocalLLM(BaseLLM):
    def __init__(
            self,
            model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
            device: str = 'cuda',
    ):
        self.model_name = model_name
        self.device = device



    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        pass

