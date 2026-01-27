import argparse
import os
from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import normalize_scores,HybridSearch
from lib.search_utils import load_movies

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
def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    norm_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a list of scores to [0, 1]"
    )
    norm_parser.add_argument(

        "scores",
        type=float,
        nargs="+",
        help="List of scores to normalize"

    ) 

    weight_parser =  subparsers.add_parser("weighted-search",help="Weight search") 
    weight_parser.add_argument("query", type=str,help="Query for weight search")

    weight_parser.add_argument("--alpha", type=float, default=0.5, 
                             help="Weight of queries per chunk (default: 0.5)")
    weight_parser.add_argument("--limit", type=int , default=5,
                              help=(" Limit of weight search (default: 5)"))

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Hybrid search using Reciprocal Rank Fusion (RRF)"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("--k", type=int, default=60,
                            help="RRF constant k (default: 60)")
    rrf_parser.add_argument("--limit", type=int, default=5,
                            help="Number of results (default: 5)")
    rrf_parser.add_argument("--enhance",type=str,choices=["spell"],
                            help="Query enhancement method",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores)
        case "weighted-search":
            print(f"Weighted hybrid search (alpha={args.alpha}): {args.query}")
            docs = load_movies()
            hybrid = HybridSearch(docs)
            results = hybrid.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['title']} (Hybrid Score: {res['score']:.4f})")
                print(f"   BM25: {res['metadata']['bm25']:.4f}, Semantic: {res['metadata']['semantic']:.4f}")
                print(f"   {res['document']}...")
        case "rrf-search":
            original_query = args.query
            final_query = original_query

            # Apply enhancement if requested
            if args.enhance == "spell":
                final_query = enhance_query_spell(original_query)
                if final_query != original_query:
                    print(f"Enhanced query (spell): '{original_query}' → '{final_query}'")
                else:
                    print("No spelling corrections needed.")

            print(f"RRF hybrid search (k={args.k}): {final_query}")
            docs = load_movies()
            hybrid = HybridSearch(docs)
            results = hybrid.rrf_search(final_query, k=args.k, limit=args.limit)

            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['title']}")
                print(f"   RRF Score: {res['score']:.4f}")
                print(f"   BM25 Rank: {res['metadata']['bm25_rank']}, "
                      f"Semantic Rank: {res['metadata']['semantic_rank']}")
                print(f"   {res['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
