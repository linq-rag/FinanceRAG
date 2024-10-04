import abc
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

__all__ = [
    "Encoder",
    "Lexical",
    "Retrieval",
    "CrossEncoder",
    "Reranker",
    "Generator",
]


class Lexical(abc.ABC):
    """
    Abstract class for lexical models that defines an interface for calculating relevance scores
    between a query and a set of documents. This abstract class is designed to be implemented by
    classes that calculate document-query relevance using lexical methods such as BM25 or
    other term-based approaches.
    """

    @abc.abstractmethod
    def get_scores(self, query: List[str], **kwargs) -> List[float]:
        """
        Calculates relevance scores for a given query against a set of documents.

        Args:
            query (`List[str]`):
                A tokenized query in the form of a list of words. This represents the query
                to be evaluated for relevance against the documents.

        Returns:
            `List[float]`:
                A list of relevance scores, where each score corresponds to the relevance of
                a document in the indexed corpus to the provided query.
        """
        raise NotImplementedError


class Encoder(abc.ABC):
    """
    Abstract class for dense encoders, providing methods to encode texts, queries, and corpora into dense vectors.
    """

    @abc.abstractmethod
    def encode_queries(
            self, queries: List[str], **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encodes a list of queries into dense vector representations.

        Args:
            queries (`List[str]`):
                A list of query strings to encode.
            **kwargs:
                Additional arguments passed to the encoder.

        Returns:
            `Union[torch.Tensor, np.ndarray]`:
                Encoded queries as a tensor or numpy array.
        """
        raise NotImplementedError

    def encode_corpus(
            self,
            corpus: Union[
                List[Dict[Literal["title", "text"], str]],
                Dict[Literal["title", "text"], List],
            ],
            **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encodes a list of corpus documents into dense vector representations.

        Args:
            corpus (`Union[List[Dict[Literal["title", "text"], str]], Dict[Literal["title", "text"], List]]`):
                A list or dictionary of corpus documents to encode.
            **kwargs:
                Additional arguments passed to the encoder.

        Returns:
            `Union[torch.Tensor, np.ndarray]`:
                Encoded corpus documents as a tensor or numpy array.
        """
        raise NotImplementedError


class Retrieval(abc.ABC):
    """
    Abstract class for retrieval modules, providing a method to search for the most relevant documents based on queries.
    """

    @abc.abstractmethod
    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            score_function: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Searches the corpus for the most relevant documents to the given queries.

        Args:
            corpus (`Dict[str, Dict[Literal["title", "text"], str]]`):
                A dictionary where each key is a document ID and each value is another dictionary containing document fields
                (e.g., {'text': str, 'title': str}).
            queries (`Dict[str, str]`):
                A dictionary where each key is a query ID and each value is the query text.
            top_k (`Optional[int]`, *optional*):
                The number of top documents to return for each query. If None, return all documents. Defaults to None.
            score_function (`Optional[str]`, *optional*):
                The scoring function to use when ranking the documents (e.g., 'cosine', 'dot', etc.). Defaults to None.
            **kwargs:
                Additional arguments passed to the search method.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where each key is a query ID, and each value is another dictionary mapping document IDs to
                relevance scores (e.g., {'doc1': 0.9, 'doc2': 0.8}).
        """
        raise NotImplementedError


class CrossEncoder(abc.ABC):
    """
    Abstract class for rerankers, providing methods to predict sentence similarity and rank documents based on queries.
    """

    @abc.abstractmethod
    def predict(
            self,
            sentences: Union[
                List[Tuple[str, str]], List[List[str]], Tuple[str, str], List[str]
            ],
            batch_size: Optional[int] = None,
            **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Predicts similarity or relevance scores for pairs of sentences or lists of sentences.

        Args:
            sentences (`Union[List[Tuple[str, str]], List[List[str]], Tuple[str, str], List[str]]`):
                Sentences to predict similarity scores for. Can be a list of sentence pairs, list of sentence lists,
                a single sentence pair, or a list of sentences.
            batch_size (`Optional[int]`, *optional*):
                Batch size for prediction. Defaults to None.

        Returns:
            `Union[torch.Tensor, np.ndarray]`:
                Predicted similarity or relevance scores as a tensor or numpy array.
        """
        raise NotImplementedError


class Reranker(abc.ABC):
    """
    Abstract class for reranking modules that defines methods to rerank search results based on queries.
    """

    @abc.abstractmethod
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
        Reranks the search results based on the given queries and the initial ranking scores.

        Args:
            corpus (`Dict[str, Dict[str, str]]`):
                A dictionary where keys are document IDs and values are dictionaries containing
                document metadata, such as content or other features.
            queries (`Dict[str, str]`):
                A dictionary where keys are query IDs and values are the corresponding query texts.
            results (`Dict[str, Dict[str, float]]`):
                A dictionary where keys are query IDs and values are dictionaries mapping document
                IDs to their initial relevance scores.
            top_k (`int`):
                The number of top documents to rerank.
            batch_size (`Optional[int]`, *optional*):
                The batch size to use during reranking. Useful for models that process data in
                batches. Defaults to None.
            **kwargs:
                Additional keyword arguments for custom configurations in the reranking process.

        Returns:
            `Dict[str, Dict[str, float]]`:
                The reranked relevance scores, returned as a dictionary mapping query IDs to dictionaries of document IDs and their scores.
        """
        raise NotImplementedError


class Generator(abc.ABC):
    """
    Abstract class for text generators, providing methods for generating text completions in a chat-like interface.
    """

    @abc.abstractmethod
    def generation(
            self, messages: Dict[str, List[Dict[str, str]]], **kwargs
    ) -> Dict[str, str]:
        """
        Generates a chat completion based on a sequence of messages.

        Args:
            messages (`Dict[str, List[Dict[str, str]]]`):
                A list of message dictionaries per `query_id`.
                Each dictionary in list must contain:
                    - 'role' (str): The role of the speaker (e.g., 'user' or 'system').
                    - 'content' (str): The content of the message.
            **kwargs:
                Additional arguments passed to the generator.

        Returns:
            `Dict[str, str]`:
                A dictionary containing the generated response, where each key is the `query_id` and the value is the generated text.
        """
        raise NotImplementedError