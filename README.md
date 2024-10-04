### (1) Environment Setup
To begin, install the necessary dependencies:

```bash
# Set up a virtual environment
python -m venv financerag_env
source financerag_env/bin/activate  # Windows: financerag_env\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

---

### (2) Folder/Class Overview

- **`retrieval/`**:
  - `DenseRetrieval`: Retrieves documents based on dense embeddings.
  - `SentenceTransformerEncoder`: Encodes queries and documents into dense vector representations.

- **`rerank/`**:
  - `CrossEncoderReranker`: Reranks retrieval results using a cross-encoder.

- **`tasks/`**:
  - `BaseTask`: A parent class of each dataset for document retrieval and ranking. Use other dataset tasks that inherit this class.

- **`generate/`**: Handles answer generation processes.

---

### (3) Example Code

1. **Initialize Dataset Task**:
   ```python
   # FinDER for example.
   # You can use other tasks such as `FinQA`, `TATQA`, etc.
   from financerag.tasks import FinDER
   finder_task = FinDER()
   ```

2. **Setup Models**:
   ```python
   from sentence_transformers import SentenceTransformer
   from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval

   model = SentenceTransformer('intfloat/e5-large-v2')
   # We need to put prefix for e5 models.
   # For more details, see Arxiv paper https://arxiv.org/abs/2212.03533
   encoder_model = SentenceTransformerEncoder(
       q_model=model,
       doc_model=model,
       query_prompt='query: ',
       doc_prompt='passage: '
   )
   retriever = DenseRetrieval(model=encoder_model)
   ```

3. **Retrieve and Rerank**:
   ```python
   # Retrieve relevant documents
   results = finder_task.retrieve(retriever=retriever)
   
   # Rerank the results
   from financerag.rerank import CrossEncoderReranker
   reranker = CrossEncoderReranker(CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2'))
   reranked_results = finder_task.rerank(reranker, results, top_k=100, batch_size=32)
   ```

4. **Save the Results**:
   After reranking, you can save the results:
   ```python
   finder_task.save_results(output_dir='path_to_save_directory')
   ```

This provides a complete workflow for initializing tasks, performing document retrieval, reranking, and saving the final results.