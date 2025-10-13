import pickle
from pathlib import Path
from typing import Tuple, List

import faiss
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from rag.embeddings import create_embedder, BaseEmbedder
from rag.config import settings
from rag.utils import batched

np_float32 = NDArray[np.float32]
np_int32 = NDArray[np.int32]

class DocumentStore:
    SUPPORTED_INDEXES = {
        "flat_ip": faiss.IndexFlatIP,
        "flat_l2": faiss.IndexFlatL2,
        "ivf_flat": faiss.IndexIVFFlat,
        # ... other supported indexes
    }

    def __init__(self, embedder: BaseEmbedder, index_name="flat_ip"):
        if index_name not in self.SUPPORTED_INDEXES:
            raise ValueError("Invalid index name '{}'".format(index_name))

        self.documents = []
        self.embedder = embedder
        self.index = self.SUPPORTED_INDEXES[index_name](embedder.dimension)

    def add_documents(self, docs: List[str], batch_size: int = 512, verbose: bool = False):
        pbar = tqdm(total=len(docs), desc='Adding documents', disable=not verbose)
        for batch in batched(docs, batch_size):
            self.documents.extend(batch)
            embeddings = self.embedder.embed_batch(batch)
            self.index.add(embeddings)
            pbar.update(len(batch))
        pbar.close()

    def search(self, query: str, k: int):
        query_emb = self.embedder.embed_batch([query])
        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, i in zip(distances.flatten(), indices.flatten()):
            results.append({
                "text": self.documents[i],
                "distance": float(dist),
                "index": int(i),
            })
        return results

    def save(self, path: str):
        # create dir
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, f"{path}.index")

        with open(f"{path}.docs", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        if not Path(f"{path}.index").exists() or (not Path(f"{path}.docs").exists()):
            raise FileNotFoundError(f"{path}.index or {path}.docs")

        with open(f"{path}.docs", "rb") as f:
            self.documents = pickle.load(f)

        self.index = faiss.read_index(f"{path}.index")



