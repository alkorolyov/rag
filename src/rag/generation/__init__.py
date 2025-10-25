from rag.config import Settings
from rag.generation.local_llm import LocalLLM
from rag.logger import setup_logger

logger = setup_logger(__name__)

def create_llm(settings: Settings):
    if settings.llm_type == "local":
        llm = LocalLLM(
            model_name=settings.llm_model,
            device=settings.llm_device,
        )
        return llm

    else:
        raise ValueError(f"Unknown llm type: {settings.llm_type}")