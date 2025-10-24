import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.generation.base import BaseLLM
from rag.logger import setup_logger

logger = setup_logger(__name__)

class LocalLLM(BaseLLM):
    def __init__(
            self,
            model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
            device: str = 'cuda',
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def generate(self, prompt: str, system_prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]
        with ctx:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        # Slice the output tokens to only include newly generated tokens
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

