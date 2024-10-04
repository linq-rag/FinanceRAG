import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from nltk.tokenize import word_tokenize

from financerag.common import Lexical, Retrieval

logger = logging.getLogger(__name__)


def tokenize_list(input_list: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of strings using the `nltk.word_tokenize` function.

    Args:
        input_list (`List[str]`):
            A list of input strings to be tokenized.

    Returns:
        `List[List[str]]`:
            A list where each element is a list of tokens corresponding to an input string.
    """
    return list(map(word_tokenize, input_list))


class BM25Retriever(Retrieval):
    """
    A retrieval class that utilizes a lexical model (e.g., BM25) to search for the most relevant documents
    from a given corpus based on the input queries. This retriever tokenizes the queries and uses the provided
    lexical model to compute relevance scores between the queries and documents in the corpus.

    Methods:
        - retrieve: Searches for relevant documents based on the given queries, returning the top-k results.
    """

    def __init__(self, model: Lexical, tokenizer: Callable[[List[str]], List[List[str]]] = tokenize_list):
        """
        Initializes the `BM25Retriever` class with a lexical model and a tokenizer function.

        Args:
            model (`Lexical`):
                A lexical model (e.g., BM25) implementing the `Lexical` protocol, responsible for calculating relevance scores.
            tokenizer (`Callable[[List[str]], List[List[str]]]`, *optional*):
                A function that tokenizes the input queries. Defaults to `tokenize_list`, which uses `nltk.word_tokenize`.
        """
        self.model: Lexical = model
        self.tokenizer: Callable[[List[str]], List[List[str]]] = tokenizer
        self.results: Optional[Dict[str, Any]] = {}

    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            score_function: Optional[str] = None,
            return_sorted: bool = False,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Searches the corpus for the most relevant documents based on the given queries. The retrieval process involves
        tokenizing the queries, calculating relevance scores using the lexical model, and returning the top-k results
        for each query.

        Args:
            corpus (`Dict[str, Dict[Literal["title", "text"], str]]`):
                A dictionary representing the corpus, where each key is a document ID, and each value is another dictionary
                containing document fields such as 'id', 'title', and 'text'.
            queries (`Dict[str, str]`):
                A dictionary containing query IDs and corresponding query texts.
            top_k (`Optional[int]`, *optional*):
                The number of top documents to return for each query. If not provided, all documents are returned. Defaults to `None`.
            return_sorted (`bool`, *optional*):
                Whether to return the results sorted by score. Defaults to `False`.
            **kwargs:
                Additional keyword arguments passed to the lexical model during scoring.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where each key is a query ID, and the value is another dictionary mapping document IDs to relevance scores.
        """
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        logger.info("Tokenizing queries with lower cases")
        query_lower_tokens = self.tokenizer([queries[qid].lower() for qid in queries])

        corpus_ids = list(corpus.keys())

        for qid, query in zip(query_ids, query_lower_tokens):
            scores = self.model.get_scores(query)
            top_k_result = np.argsort(scores)[::-1][:top_k]
            for idx in top_k_result:
                self.results[qid][corpus_ids[idx]] = scores[idx]

        return self.results