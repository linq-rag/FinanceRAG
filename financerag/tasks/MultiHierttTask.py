from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class MultiHiertt(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="MultiHiertt",
            description="MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data",
            reference="https://github.com/psunlpgroup/MultiHiertt",
            dataset={
                "path": "Linq-AI-Research/FinanceRAG",
                "subset": "MultiHiertt",
            },
            type="RAG",
            category="s2p",
            modalities=["text"],
            date=None,
            domains=["Report"],
            task_subtypes=[
                "Financial retrieval",
                "Question answering",
            ],
            license="mit",
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="human-generated",
            bibtex_citation="""
                @inproceedings{zhao-etal-2022-multihiertt,
                    title = "{M}ulti{H}iertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data",
                    author = "Zhao, Yilun  and
                      Li, Yunxiang  and
                      Li, Chenying  and
                      Zhang, Rui",
                    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
                    month = may,
                    year = "2022",
                    address = "Dublin, Ireland",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/2022.acl-long.454",
                    pages = "6588--6600",
                }
            """,
        )
        super().__init__(self.metadata)