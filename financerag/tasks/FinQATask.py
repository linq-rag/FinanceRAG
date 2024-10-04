from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class FinQA(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="FinQA",
            description="FinQA: A Dataset of Numerical Reasoning over Financial Data",
            reference="https://github.com/czyssrs/FinQA",
            dataset={
                "path": "Linq-AI-Research/FinanceRAG",
                "subset": "FinQA",
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
                @article{chen2021finqa,
                    title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
                    author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
                    journal={Proceedings of EMNLP 2021},
                    year={2021}
                }
            """,
        )
        super().__init__(self.metadata)
