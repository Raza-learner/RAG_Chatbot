import argparse
import os
from dotenv import load_dotenv
from google import genai
from lib.hybrid_search import (normalize_scores,
                                HybridSearch, 
                                enhance_query_spell,
                                enhance_query_rewrite,
                                enhance_query_expand)
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
    rrf_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],
                            help="Query enhancement method",)
    rrf_parser.add_argument("--rerank-method",type=str,choices=["individual","batch","cross_encoder"],
                            help="Reranking method after RRF (individual = LLM per doc)",)
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
            
            if args.enhance:
                if args.enhance == "spell":
                    final_query = enhance_query_spell(original_query)
                    method = "spell"
                elif args.enhance == "rewrite":
                    final_query = enhance_query_rewrite(original_query)
                    method = "rewrite"
                elif args.enhance == "expand":
                    final_query = enhance_query_expand(original_query)
                    method = "expand"

                if final_query != original_query:
                    print(f"Enhanced query ({method}): '{original_query}' → '{final_query}'")
                else:
                    print(f"No enhancement needed ({method}).")

            print(f"RRF hybrid search (k={args.k}): {final_query}")
            docs = load_movies()
            hybrid = HybridSearch(docs)
            results = hybrid.rrf_search(
                final_query,
                k=args.k,
                limit=args.limit,
                rerank_method=args.rerank_method
            )

            for i, res in enumerate(results, 1):
                print(f"\n{i}. {res['title']}")
                print(f"   RRF Score: {res['score']:.4f}")
                if "rerank_score" in res:
                    print(f"   Rerank Score: {res['rerank_score']:.1f}/10")
                print(f"   BM25 Rank: {res['metadata']['bm25_rank']}, "
                      f"Semantic Rank: {res['metadata']['semantic_rank']}")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
