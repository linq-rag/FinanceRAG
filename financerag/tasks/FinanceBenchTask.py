from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class FinanceBench(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="FinanceBench",
            description="FinanceBench: A New Benchmark for Financial Question Answering",
            reference="https://github.com/patronus-ai/financebench",
            dataset={
                "path": "Linq-AI-Research/FinanceRAG",
                "subset": "FinanceBench",
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
            license=None,
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="human-generated",
            bibtex_citation="""
                @misc{islam2023financebench,
                      title={FinanceBench: A New Benchmark for Financial Question Answering},
                      author={Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
                      year={2023},
                      eprint={2311.11944},
                      archivePrefix={arXiv},
                      primaryClass={cs.CL}
                }
            """,
        )
        super().__init__(metadata=self.metadata)