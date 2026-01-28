import os
from dotenv import load_dotenv
from google import genai

from sentence_transformers import CrossEncoder

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.document_map = {doc["id"]: doc for doc in documents}
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
        """
        Hybrid search: combine BM25 and chunked semantic scores with alpha weighting.
        Returns top `limit` movies with hybrid scores.
        """
        # 1. Get lots of candidates (500× limit for good coverage)
        bm25_results = self._bm25_search(query, limit=500 * limit)
        sem_results = self.semantic_search.search_chunks(query, limit=500 * limit)

        # 2. Normalize BM25 scores (0–1)
        bm25_scores = [res["score"] for res in bm25_results]
        bm25_norm = normalize_scores(bm25_scores) if bm25_scores else []
        bm25_norm_dict = {bm25_results[i]["id"]: bm25_norm[i] 
                         for i in range(len(bm25_norm))}

        # 3. Normalize semantic scores (0–1)
        sem_scores = [res["score"] for res in sem_results]
        sem_norm = normalize_scores(sem_scores) if sem_scores else []
        sem_norm_dict = {sem_results[i]["id"]: sem_norm[i] 
                        for i in range(len(sem_norm))}

        # 4. Collect all unique movie IDs and compute hybrid score
        all_movie_ids = set(bm25_norm_dict.keys()) | set(sem_norm_dict.keys())
        hybrid_scores = {}

        for movie_id in all_movie_ids:
            bm25_normed = bm25_norm_dict.get(movie_id, 0.0)
            sem_normed = sem_norm_dict.get(movie_id, 0.0)
            hybrid = alpha * bm25_normed + (1 - alpha) * sem_normed
            hybrid_scores[movie_id] = hybrid

        # 5. Sort by hybrid score descending
        sorted_movies = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 6. Build final formatted results
        top_results = []
        for movie_id, hybrid_score in sorted_movies[:limit]:
            movie = self.document_map.get(movie_id)   # ← FIXED HERE
            if movie is None:
                continue

            top_results.append({
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "score": round(hybrid_score, 4),
                "metadata": {
                    "bm25": round(bm25_norm_dict.get(movie_id, 0.0), 4),
                    "semantic": round(sem_norm_dict.get(movie_id, 0.0), 4)
                }
            })

        
        return top_results

    def rrf_search(self, query: str, k: int = 60, limit: int = 10, rerank_method: str = None) -> list[dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        Combines BM25 and semantic chunk rankings.
        """
        # Fetch more candidates if reranking
        fetch_limit = limit * 5 if rerank_method else limit

        bm25_results = self._bm25_search(query, limit=fetch_limit)
        sem_results = self.semantic_search.search_chunks(query, limit=fetch_limit)

        # 2. Build movie → rank dicts (rank starts from 1 = best)
        bm25_rank = {}
        for rank, res in enumerate(bm25_results, 1):
            movie_id = res["id"]
            bm25_rank[movie_id] = rank

        sem_rank = {}
        for rank, res in enumerate(sem_results, 1):
            movie_id = res["id"]
            sem_rank[movie_id] = rank

        # 3. Compute RRF score for every movie that appears in at least one list
        all_movie_ids = set(bm25_rank.keys()) | set(sem_rank.keys())
        rrf_scores = {}

        for movie_id in all_movie_ids:
            rank_bm25 = bm25_rank.get(movie_id, float('inf'))  # inf if not present
            rank_sem = sem_rank.get(movie_id, float('inf'))
            score = rrf_score(rank_bm25, k) + rrf_score(rank_sem, k)
            rrf_scores[movie_id] = score

        # 4. Sort by RRF score descending
        sorted_movies = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 5. Build formatted results
        initial_results = []
        for movie_id, rrf_score_val in sorted_movies:
            movie = self.document_map.get(movie_id)
            if movie is None:
                continue

            bm25_r = bm25_rank.get(movie_id, "N/A")
            sem_r = sem_rank.get(movie_id, "N/A")

            initial_results.append({
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "score": round(rrf_score_val, 4),
                "metadata": {
                    "bm25_rank": bm25_r,
                    "semantic_rank": sem_r
                }
            })

        # Apply rerank if requested
        if rerank_method == "individual":
            print(f"Reranking top {len(initial_results)} results using individual method...")
            return llm_rerank(query, initial_results, limit)

        elif rerank_method == "batch":
            print(f"Reranking top {len(initial_results)} results using batch method...")
            return batch_llm_rerank(query, initial_results, limit)
        elif rerank_method == "cross_encoder":
            print(f"Reranking top {len(initial_results)} results using cross_encoder method...")
            return cross_encoder_rerank(query, initial_results, limit)
    
        return initial_results[:limit]        

def normalize_scores(scores: list[float]) -> list[float]:
    """
    Return list of scores normalized to [0, 1] using min-max.
    Returns empty list if input empty.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    denom = max_score - min_score
    return [(s - min_score) / denom for s in scores]

def rrf_score(rank: int, k: int = 60) -> float:
    """
    Reciprocal Rank Fusion formula: 1 / (k + rank)
    """
    return 1.0 / (k + rank)

def enhance_query_spell(query: str) -> str:
    """Use Gemini to correct spelling in the query."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found → skipping spell correction")
        return query

    client = genai.Client(api_key=api_key)

    prompt = f"""Fix any spelling errors in this movie search query.
    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        
        corrected = response.text.strip()
        
       
        if corrected.lower().startswith("corrected:"):
            corrected = corrected.split(":", 1)[1].strip()
            
       
        corrected = corrected.strip('"')
        
        return corrected
    except Exception as e:
        print(f"Spell correction failed: {e} → using original query")
        
        return query

def enhance_query_rewrite(query: str) -> str:
    """Use Gemini to correct spelling in the query."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found → skipping spell correction")
        return query

    client = genai.Client(api_key=api_key)

    prompt =f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        
        corrected = response.text.strip()
        
       
        if corrected.lower().startswith("corrected:"):
            corrected = corrected.split(":", 1)[1].strip()
            
       
        corrected = corrected.strip('"')
        
        return corrected
    except Exception as e:
        print(f"Spell correction failed: {e} → using original query")
        
        return query

def enhance_query_expand(query: str) -> str:
    """Use Gemini to correct spelling in the query."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found → skipping spell correction")
        return query

    client = genai.Client(api_key=api_key)

    prompt =f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}" """    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        
        corrected = response.text.strip()
        
       
        if corrected.lower().startswith("corrected:"):
            corrected = corrected.split(":", 1)[1].strip()
            
       
        corrected = corrected.strip('"')
        
        return corrected
    except Exception as e:
        print(f"Spell correction failed: {e} → using original query")
        
        return query


def llm_rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Rerank top results using Gemini individual scoring."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY missing → skipping LLM rerank")
        return results[:limit]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Safety settings to allow more content (optional but helps with movie data)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    reranked = []
    for res in results:
        doc_text = f"{res['title']} - {res['document']}"
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc_text}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        try:
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings
            )
            score_text = response.text.strip()
            score = float(score_text) if score_text.isdigit() or '.' in score_text else 0.0
        except Exception as e:
            print(f"LLM rerank failed for {res['title']}: {e}")
            score = 0.0

        reranked.append({**res, "rerank_score": score})

        time.sleep(3)  # avoid rate limits

    # Sort by new LLM score descending
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:limit]

def batch_llm_rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Rerank top results using a single Gemini batch prompt."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY missing → skipping batch rerank")
        return results[:limit]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Build document list string
    doc_list_str = ""
    id_to_result = {}
    for i, res in enumerate(results, 1):
        doc_id = res["id"]
        id_to_result[doc_id] = res
        doc_list_str += f"ID {doc_id}: {res['title']} - {res['document']}\n"

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        # Extract JSON list (remove extra text if any)
        if response_text.startswith('[') and response_text.endswith(']'):
            ranked_ids = json.loads(response_text)
        else:
            ranked_ids = json.loads(response_text.split('[', 1)[1].rsplit(']', 1)[0])

        # Map back to original results
        reranked = []
        for movie_id in ranked_ids:
            if movie_id in id_to_result:
                reranked.append(id_to_result[movie_id])

        return reranked[:limit]
    except Exception as e:
        print(f"Batch rerank failed: {e} → using original RRF order")
        return results[:limit]

def cross_encoder_rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Rerank results using a cross-encoder model."""
    if not results:
        return []

    # Prepare pairs: [query, doc_title - doc_text]
    pairs = []
    for res in results:
        doc_text = f"{res['title']} - {res['document']}"
        pairs.append([query, doc_text])

    try:
        # Load cross-encoder once (tiny and fast)
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

        # Predict scores (higher = better relevance)
        scores = cross_encoder.predict(pairs)

        # Attach scores to results
        reranked = []
        for res, score in zip(results, scores):
            reranked.append({**res, "rerank_score": float(score)})

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:limit]

    except Exception as e:
        print(f"Cross-encoder rerank failed: {e} → using original RRF order")
        return results[:limit]
