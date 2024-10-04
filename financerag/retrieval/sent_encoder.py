from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor

from financerag.common import Encoder


# Adopted by https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/sentence_bert.py
class SentenceTransformerEncoder(Encoder):

    def __init__(
            self,
            model_name_or_path: Union[str, Tuple[str, str]],
            query_prompt: Optional[str] = None,
            doc_prompt: Optional[str] = None,
            **kwargs
    ):
        if isinstance(model_name_or_path, str):
            self.q_model = SentenceTransformer(model_name_or_path, **kwargs)
            self.doc_model = self.q_model
        elif isinstance(model_name_or_path, Tuple):
            self.q_model = SentenceTransformer(model_name_or_path[0], **kwargs)
            self.doc_model = SentenceTransformer(model_name_or_path[1], **kwargs)
        else:
            raise TypeError
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def encode_queries(
            self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[np.ndarray, Tensor]:
        if self.query_prompt is not None:
            queries = [self.query_prompt + query for query in queries]
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
            self,
            corpus: Union[
                List[Dict[Literal["title", "text"], str]],
                Dict[Literal["title", "text"], List],
            ],
            batch_size: int = 8,
            **kwargs
    ) -> Union[np.ndarray, Tensor]:
        if isinstance(corpus, dict):
            sentences = [
                (
                    (corpus["title"][i] + " " + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                )
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (
                    (doc["title"] + " " + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                )
                for doc in corpus
            ]
        if self.doc_prompt is not None:
            sentences = [self.doc_prompt + s for s in sentences]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)
