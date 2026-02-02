# RAG Search Engine

A Retrieval-Augmented Generation (RAG) search engine for movies, featuring hybrid search (BM25 + semantic), query enhancement, reranking, and LLM-powered responses using Google's Gemini API.

## Features

- **Keyword Search** - BM25 ranking with TF-IDF scoring and inverted index
- **Semantic Search** - Dense vector search using sentence-transformers
- **Hybrid Search** - Combines keyword and semantic search via:
  - Weighted score fusion
  - Reciprocal Rank Fusion (RRF)
- **Query Enhancement** - LLM-powered spell correction, query rewriting, and expansion
- **Reranking** - Multiple reranking strategies:
  - Individual LLM scoring
  - Batch LLM ranking
  - Cross-encoder reranking
- **RAG Generation** - Answer questions using retrieved context
- **Multimodal Queries** - Combine image + text for query rewriting
- **Evaluation** - Precision@k, Recall@k, and F1 scoring with golden dataset

## Installation

Requires Python 3.13+

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-search-engine.git
cd rag-search-engine

# Install dependencies with uv
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Keyword Search

```bash
# Build inverted index
uv run cli/keyword_search_cli.py build

# Search with BM25
uv run cli/keyword_search_cli.py bm25search "space adventure"

# Get TF-IDF scores
uv run cli/keyword_search_cli.py tfidf 1 "robot"
```

### Semantic Search

```bash
# Verify embedding model
uv run cli/semantic_search_cli.py verify

# Generate embeddings for chunks
uv run cli/semantic_search_cli.py embed-chunks

# Search with semantic similarity
uv run cli/semantic_search_cli.py search "emotional drama about family"

# Search using chunked embeddings
uv run cli/semantic_search_cli.py search-chunked "time travel paradox"
```

### Hybrid Search

```bash
# Weighted hybrid search (alpha controls BM25 vs semantic balance)
uv run cli/hybrid_search_cli.py weighted-search "funny comedy" --alpha 0.5

# RRF hybrid search
uv run cli/hybrid_search_cli.py rrf-search "thriller with twist ending"

# With query enhancement
uv run cli/hybrid_search_cli.py rrf-search "moive abuot space" --enhance spell
uv run cli/hybrid_search_cli.py rrf-search "that bear movie" --enhance rewrite
uv run cli/hybrid_search_cli.py rrf-search "scary film" --enhance expand

# With reranking
uv run cli/hybrid_search_cli.py rrf-search "action movie" --rerank-method cross_encoder
```

### RAG (Retrieval-Augmented Generation)

```bash
# Basic RAG query
uv run cli/augmented_generation_cli.py rag "recommend me a sci-fi movie"

# Summarize search results
uv run cli/augmented_generation_cli.py summarize "horror movies" --limit 5

# Answer questions
uv run cli/augmented_generation_cli.py question "What are some good movies for family movie night?"
```

### Multimodal Query Rewriting

```bash
# Rewrite query based on image content
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "cute bear movie"
```

### Evaluation

```bash
# Evaluate search quality against golden dataset
uv run cli/evaluation_cli.py --limit 5
```

## Project Structure

```
├── cli/
│   ├── lib/
│   │   ├── hybrid_search.py    # Hybrid search + reranking
│   │   ├── keyword_search.py   # BM25 + inverted index
│   │   ├── semantic_search.py  # Embedding-based search
│   │   └── search_utils.py     # Shared utilities
│   ├── augmented_generation_cli.py
│   ├── describe_image_cli.py
│   ├── evaluation_cli.py
│   ├── hybrid_search_cli.py
│   ├── keyword_search_cli.py
│   └── semantic_search_cli.py
├── data/
│   ├── movies.json             # Movie dataset
│   ├── golden_dataset.json     # Evaluation test cases
│   └── stopwords.txt
├── pyproject.toml
└── README.md
```

## How It Works

### Search Pipeline

1. **Query Enhancement** (optional) - Correct spelling, rewrite, or expand the query using Gemini
2. **Dual Retrieval** - Run both BM25 (keyword) and semantic search in parallel
3. **Score Fusion** - Combine results using weighted average or RRF
4. **Reranking** (optional) - Refine rankings using cross-encoder or LLM
5. **Generation** (RAG only) - Generate response using retrieved context

### Reciprocal Rank Fusion (RRF)

RRF combines rankings from multiple retrieval methods:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Where `k` is a constant (default 60) and `rank_i(d)` is the rank of document `d` in retriever `i`.

## Dependencies

- `google-genai` - Gemini API client
- `sentence-transformers` - Semantic embeddings
- `nltk` - Text preprocessing
- `numpy` - Numerical operations
- `pillow` - Image processing
- `python-dotenv` - Environment configuration

## License

MIT
