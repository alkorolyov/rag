from rag.config import Settings
from rag.generation.local_llm import LocalLLM

def create_llm(settings: Settings):
    if settings.llm_type == "local":
        return LocalLLM(
            model_name=settings.llm_model,
            device=settings.llm_device,
        )
    else:
        raise ValueError(f"Unknown llm type: {settings.llm_type}")