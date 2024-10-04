import logging
import multiprocessing
import os
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, cast

import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from financerag.common.protocols import Generator

openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


class OpenAIGenerator(Generator):
    """
    A class that interfaces with the OpenAI API to generate responses using a specified model. It implements the
    `Generator` protocol and supports generating responses in parallel using multiple processes.

    Args:
        model_name (`str`):
            The name of the OpenAI model to use for generating responses (e.g., "gpt-4", "gpt-3.5-turbo").
    """

    def __init__(self, model_name: str):
        """
        Initializes the OpenAIGenerator with the specified model name.

        Args:
            model_name (`str`):
                The OpenAI model name used to generate responses.
        """
        self.model_name: str = model_name
        self.results: Dict = {}

    def _process_query(
            self, args: Tuple[str, List[ChatCompletionMessageParam], Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Internal method to process a single query using the OpenAI model. It sends the query and messages to the
        OpenAI API and retrieves the response.

        Args:
            args (`Tuple[str, List[ChatCompletionMessageParam], Dict[str, Any]]`):
                Contains the query ID, a list of messages (query), and additional arguments for the model.

        Returns:
            `Tuple[str, str]`:
                A tuple containing the query ID and the generated response.
        """
        q_id, messages, kwargs = args
        temperature = kwargs.pop("temperature", 1.0)
        top_p = kwargs.pop("top_p", 1.0)
        stream = kwargs.pop("stream", False)
        max_tokens = kwargs.pop("max_tokens", 10000)
        presence_penalty = kwargs.pop("presence_penalty", 0.0)
        frequency_penalty = kwargs.pop("frequency_penalty", 0.0)

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        return q_id, response.choices[0].message.content

    def generation(
            self,
            messages: Dict[str, List[Dict[str, str]]],
            num_processes: int = multiprocessing.cpu_count(),  # Number of parallel processes
            **kwargs,
    ) -> Dict[str, str]:
        """
        Generate responses for the given messages using the OpenAI model. This method supports parallel processing
        using multiprocessing to speed up the generation process for multiple queries.

        Args:
            messages (`Dict[str, List[Dict[str, str]]]`):
                A dictionary where the keys are query IDs, and the values are lists of dictionaries representing the
                messages (queries).
            num_processes (`int`, *optional*, defaults to `multiprocessing.cpu_count()`):
                The number of processes to use for parallel generation.
            **kwargs:
                Additional keyword arguments for the OpenAI model (e.g., temperature, top_p, max_tokens).

        Returns:
            `Dict[str, str]`:
                A dictionary where each key is a query ID, and the value is the generated response.
        """
        logger.info(
            f"Starting generation for {len(messages)} queries using {num_processes} processes..."
        )

        # Prepare arguments for multiprocessing
        query_args = [
            (q_id, cast(list[ChatCompletionMessageParam], msg), kwargs.copy())
            for q_id, msg in messages.items()
        ]

        # Use multiprocessing Pool for parallel generation
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._process_query, query_args)

        # Collect results
        self.results = {q_id: content for q_id, content in results}

        logger.info(
            f"Generation completed for all queries. Collected {len(self.results)} results."
        )

        return self.results