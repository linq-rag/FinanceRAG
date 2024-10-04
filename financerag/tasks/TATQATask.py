from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class TATQA(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="TAT-QA",
            description="TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance",
            reference="https://github.com/NExTplusplus/TAT-QA",
            dataset={
                "path": "Linq-AI-Research/FinanceRAG",
                "subset": "TATQA",
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
            annotations_creators="human-annotated",
            dialect=[],
            sample_creation="human-generated",
            bibtex_citation="""
                @inproceedings{zhu-etal-2021-tat,
                    title = "{TAT}-{QA}: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance",
                    author = "Zhu, Fengbin  and
                      Lei, Wenqiang  and
                      Huang, Youcheng  and
                      Wang, Chao  and
                      Zhang, Shuo  and
                      Lv, Jiancheng  and
                      Feng, Fuli  and
                      Chua, Tat-Seng",
                    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
                    month = aug,
                    year = "2021",
                    address = "Online",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/2021.acl-long.254",
                    doi = "10.18653/v1/2021.acl-long.254",
                    pages = "3277--3287"
                }
            """,
        )
        super().__init__(self.metadata)
