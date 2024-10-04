import logging
from typing import Dict, Optional

from financerag.common import CrossEncoder, Reranker

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py
class CrossEncoderReranker(Reranker):
    """
    A reranker class that utilizes a cross-encoder model from the `sentence-transformers` library
    to rerank search results based on query-document pairs. This class implements a reranking
    mechanism using cross-attention, where each query-document pair is passed through the
    cross-encoder model to compute relevance scores.

    The cross-encoder model expects two inputs (query and document) and directly computes a
    score indicating the relevance of the document to the query. The model follows the
    `CrossEncoder` protocol, ensuring it is compatible with `sentence-transformers` cross-encoder models.

    Methods:
        rerank:
            Takes in a corpus, queries, and initial retrieval results, and reranks
            the top-k documents using the cross-encoder model.
    """

    def __init__(self, model: CrossEncoder):
        """
        Initializes the `CrossEncoderReranker` class with a cross-encoder model.

        Args:
            model (`CrossEncoder`):
                A cross-encoder model implementing the `CrossEncoder` protocol from the `sentence-transformers` library.
        """
        self.model: CrossEncoder = model
        self.results: Dict[str, Dict[str, float]] = {}

    def rerank(
            self,
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str],
            results: Dict[str, Dict[str, float]],
            top_k: int,
            batch_size: Optional[int] = None,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Reranks the top-k documents for each query based on cross-encoder model predictions.

        Args:
            corpus (`Dict[str, Dict[str, str]]`):
                A dictionary representing the corpus, where each key is a document ID and each value is a dictionary
                containing the title and text fields of the document.
            queries (`Dict[str, str]`):
                A dictionary containing query IDs as keys and the corresponding query texts as values.
            results (`Dict[str, Dict[str, float]]`):
                A dictionary containing query IDs and the initial retrieval results. Each query ID is mapped to another
                dictionary where document IDs are keys and initial retrieval scores are values.
            top_k (`int`):
                The number of top documents to rerank for each query.
            batch_size (`Optional[int]`, *optional*):
                The batch size used when passing the query-document pairs through the cross-encoder model.
                Defaults to None.
            **kwargs:
                Additional arguments passed to the cross-encoder model during prediction.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary containing query IDs as keys and dictionaries of reranked document IDs and their scores as values.
        """
        sentence_pairs, pair_ids = [], []

        for query_id in results:
            if len(results[query_id]) > top_k:
                for doc_id, _ in sorted(
                        results[query_id].items(), key=lambda item: item[1], reverse=True
                )[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (
                            corpus[doc_id].get("title", "")
                            + " "
                            + corpus[doc_id].get("text", "")
                    ).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (
                            corpus[doc_id].get("title", "")
                            + " "
                            + corpus[doc_id].get("text", "")
                    ).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

        #### Starting to Rerank using cross-attention
        logger.info(f"Starting To Rerank Top-{top_k}....")
        rerank_scores = [
            float(score)
            for score in self.model.predict(
                sentences=sentence_pairs, batch_size=batch_size, **kwargs
            )
        ]

        #### Reranker results
        self.results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.results[query_id][doc_id] = score

        return self.results