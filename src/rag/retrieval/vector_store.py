import pickle
from pathlib import Path
from typing import Tuple, List, Any, Optional, Sequence

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

        self.index_name = index_name

        self.documents = {}  # Store as dict: {id: text}
        self.embedder = embedder

        base_index = self.SUPPORTED_INDEXES[index_name](embedder.dimension)
        self.index = faiss.IndexIDMap(base_index)


    def add_documents(self, docs: Sequence[str], ids: Sequence[int], batch_size: int = 512, verbose: bool = False):
        if len(ids) != len(docs):
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of documents ({len(docs)})")

        pbar = tqdm(total=len(docs), desc='Adding documents', disable=not verbose)
        for i, batch in enumerate(batched(docs, batch_size)):
            batch_start_idx = i * batch_size
            batch_end_idx = batch_start_idx + len(batch)

            batch_ids_list = ids[batch_start_idx:batch_end_idx]

            # Store documents in dict keyed by ID
            for doc_id, doc_text in zip(batch_ids_list, batch):
                self.documents[doc_id] = doc_text

            embeddings = self.embedder.embed_batch(batch)

            batch_ids = np.array(batch_ids_list, dtype=np.int64)
            self.index.add_with_ids(embeddings, batch_ids)

            pbar.update(len(batch))
        pbar.close()

    def search(self, query: str, k: int):
        query_emb = self.embedder.embed_batch([query])
        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, doc_id in zip(distances.flatten(), indices.flatten()):
            doc_id = int(doc_id)
            if doc_id in self.documents:
                results.append({
                    "text": self.documents[doc_id],
                    "distance": float(dist),
                    "id": doc_id,
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

    def __repr__(self):
        return f"{self.__class__.__name__}(total={self.index.ntotal}, index={self.index_name})"