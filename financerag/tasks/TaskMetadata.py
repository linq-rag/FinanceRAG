# Adapted from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/TaskMetadata.py
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional, Union

from pydantic import AnyUrl, BaseModel, BeforeValidator, TypeAdapter, field_validator
from typing_extensions import Annotated, Literal

TASK_SUBTYPE = Literal[
    "Financial retrieval",
    "Question answering",
]

TASK_DOMAIN = Literal["Report",]

SAMPLE_CREATION_METHOD = Literal[
    "found",
    "human-generated",
    "LM-generated and verified",
]

TASK_TYPE = Literal["RAG",]

TASK_CATEGORY = Literal["s2p",]  # Sentence-to-paragraph

ANNOTATOR_TYPE = Literal[
    "expert-annotated",
    "human-annotated",
    "derived",
    "LM-generated",
    "LM-generated and reviewed",  # reviewed by humans
]

http_url_adapter = TypeAdapter(AnyUrl)
STR_URL = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a URL

pastdate_adapter = TypeAdapter(date)
STR_DATE = Annotated[
    str, BeforeValidator(lambda value: str(pastdate_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a valid date

SPLIT_NAME = str
HFSubset = str

LICENSES = (  # this list can be extended as needed
    Literal[  # we use lowercase for the licenses similar to the huggingface datasets
        "not specified",  # or none found
        "mit",
        "cc-by-2.0",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
        "cc-by-nc-4.0",
        "cc-by-nc-sa-3.0",
        "cc-by-nc-sa-4.0",
        "cc-by-nc-nd-4.0",
        "openrail",
        "openrail++",
        "odc-by",
        "afl-3.0",
        "apache-2.0",
        "cc-by-nd-2.1-jp",
        "cc0-1.0",
        "bsd-3-clause",
        "gpl-3.0",
        "cdla-sharing-1.0",
        "mpl-2.0",
    ]
)

METRIC_NAME = str
METRIC_VALUE = Union[int, float, Dict[str, Any]]

logger = logging.getLogger(__name__)


class TaskMetadata(BaseModel):
    """
    Metadata for a task.

    Args:
        dataset: A dictionary containing the arguments to pass to `datasets.load_dataset` to load the dataset for the task. Must include 'path' and *Optional*ly 'revision'.
                 Refer to https://huggingface.co/docs/datasets/v3.0.0/en/package_reference/loading_methods for more details.
        name: The name of the task.
        description: A description of the task.
        type: (*Optional*) The type of the task, such as "Retrieval" or "Generation". Corresponds to the TASK_TYPE literal.
        modalities: The input modality of the dataset. In this case, it is set to ["text"], meaning the dataset deals with textual data.
        category: (*Optional*) The category of the task, e.g., "s2p" (sentence-to-paragraph). Corresponds to the TASK_CATEGORY literal.
        reference: (*Optional*) A URL to documentation or a published paper about the task. Must be a valid URL.
        date: (*Optional*) A tuple containing the start and end dates when the dataset was collected, ensuring the data reflects a certain time frame.
        domains: (*Optional*) The domain(s) of the data, e.g., "Report". Defined as TASK_DOMAIN literals.
        task_subtypes: (*Optional*) Subtypes of the task, providing more specific details (e.g., "Financial retrieval", "Question answering").
        license: (*Optional*) The license under which the dataset is released. Uses a predefined list of licenses (e.g., "cc-by-4.0"), but custom licenses can be provided via URLs.
        annotations_creators: (*Optional*) The type of annotators who created or verified the dataset annotations, such as "expert-annotated" or "LM-generated and reviewed".
        dialect: (*Optional*) The dialect of the data, if applicable. Ideally specified as a BCP-47 language tag. Empty if no dialects are present.
        sample_creation: (*Optional*) The method used to create the dataset samples, such as "found", "human-generated", or "LM-generated and verified".
        bibtex_citation: (*Optional*) The BibTeX citation for the dataset. Should be provided if available; otherwise, it is an empty string.

    Methods:
        validate_metadata: Validates that the necessary metadata fields (like dataset path and revision) are specified.
        is_filled: Checks if all required metadata fields are filled in the TaskMetadata instance.
        intext_citation: Generates an in-text citation based on the BibTeX entry provided. If no BibTeX is available, returns an empty string.

    Validators:
        _check_dataset_path_is_specified: Ensures that the dataset dictionary contains the 'path' key.
        _check_dataset_revision_is_specified: Ensures that the dataset dictionary contains the 'revision' key or provides a warning if it's missing.
    """

    dataset: dict

    name: str
    description: str
    type: Optional[TASK_TYPE] = None
    modalities: list[Literal["text"]] = ["text"]
    category: Optional[TASK_CATEGORY] = None
    reference: Optional[STR_URL] = None

    date: Optional[tuple[STR_DATE, STR_DATE]] = None
    domains: Optional[list[TASK_DOMAIN]] = None
    task_subtypes: Optional[list[TASK_SUBTYPE]] = None
    license: Optional[LICENSES | STR_URL] = None

    annotations_creators: Optional[ANNOTATOR_TYPE] = None
    dialect: Optional[list[str]] = None

    sample_creation: Optional[SAMPLE_CREATION_METHOD] = None
    bibtex_citation: Optional[str] = None

    @field_validator("dataset")
    def _check_dataset_path_is_specified(
        cls, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        if "path" not in dataset:
            raise ValueError("Dataset path must be specified")
        return dataset

    @field_validator("dataset")
    def _check_dataset_subset_is_specified(
        cls, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        if "subset" not in dataset:
            raise ValueError("Dataset subset must be specified")
        return dataset

    def is_filled(self) -> bool:
        """Check if all the metadata fields are filled."""
        return all(
            getattr(self, field_name) is not None for field_name in self.model_fields
        )

    @property
    def intext_citation(self, include_cite: bool = True) -> str:
        """Create an in-text citation for the dataset."""
        cite = ""
        if self.bibtex_citation:
            cite = f"{self.bibtex_citation.split(',')[0].split('{')[1]}"
        if include_cite and cite:
            # check for whitespace in the citation
            if " " in cite:
                logger.warning(
                    "Citation contains whitespace. Please ensure that the citation is correctly formatted."
                )
            return f"\\cite{{{cite}}}"
        return cite
