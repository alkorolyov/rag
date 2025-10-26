from typing import Optional

import numpy as np
import redis
from rag.config import Settings
from random import uniform

class EmbeddingCache:
    def __init__(self, settings: Settings):
        self.redis = redis.from_url(settings.redis_url)
        self.settings = settings

    def get(self, text: str) -> Optional[np.ndarray]:
        key = hash(text)
        emb_bytes = self.redis.get(key)
        if emb_bytes is None:
            return None
        return np.frombuffer(emb_bytes, dtype=np.float32)

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = hash(text)
        keep_time = int(self.settings.redis_ttl * uniform(0.9, 1.1))
        self.redis.setex(key, keep_time, embedding.astype(np.float32).tobytes())