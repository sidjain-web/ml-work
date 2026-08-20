# rag_mini

A minimal **Retrieval-Augmented Generation** pipeline in pure NumPy — the
retrieval half of RAG with no external vector DB or API.

## Files
- `rag.py` — `embed`, `VectorStore` (add / cosine search), `build_prompt`,
  and an `answer` function whose generation step is a clearly-marked stub.

## Run
```bash
python rag.py
```

## Pipeline
1. **Embed** each document into a fixed-dim vector (hashing bag-of-words here;
   swap in real sentence embeddings for production quality).
2. **Store** the unit-normalized vectors in a matrix.
3. **Retrieve** the top-k documents for a query by cosine similarity.
4. **Assemble** a grounded prompt from the retrieved context.
5. **Generate** — plug in an LLM (the `nano_gpt` model here, or an API model).

The embedding is deterministic (MD5-based hashing), so results are reproducible
across runs. The quality ceiling is the embedding: replace `embed()` first when
adapting this to real use.
