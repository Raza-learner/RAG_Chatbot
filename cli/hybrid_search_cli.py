import argparse

from lib.hybrid_search import normalize_scores,HybridSearch
from lib.search_utils import load_movies
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
            print(f"RRF hybrid search (k={args.k}): {args.query}")
            docs = load_movies()
            hybrid = HybridSearch(docs)
            results = hybrid.rrf_search(args.query, k=args.k, limit=args.limit)
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
