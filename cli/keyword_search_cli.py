#!/usr/bin/env python3

import argparse
from lib.search_utils import BM25_K1,BM25_B,DEFAULT_SEARCH_LIMIT

from lib.keyword_search import (
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    InvertedIndex,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a given term"
    )
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given document ID and term"
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 saturated + length-normalized term frequency"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term")
    bm25_tf_parser.add_argument("--k1", type=float, default=BM25_K1,
                                help=f"k1 parameter (default: {BM25_K1})")
    bm25_tf_parser.add_argument("--b", type=float, default=BM25_B,
                                help=f"b parameter - length normalization (default: {BM25_B})")
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
        help=f"Number of results to return (default: {DEFAULT_SEARCH_LIMIT})"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf_value = bm25_tf_command(args.doc_id,args.term,k1=args.k1,b=args.b)
            raw_tf = tf_command(args.doc_id, args.term)
            print(
                f"BM25 TF (k1={args.k1:.2f}, b={args.b:.2f}): {bm25tf_value:.2f}")

        case "bm25search":
            print("Searching with BM25 for:", args.query)
            idx = InvertedIndex()
            idx.load()
            results = idx.bm25_search(args.query, limit=args.limit)
            if not results:
                print("No results found.")
            else:
                for i, res in enumerate(results, 1):
                    print(f"{i}. (score: {res['score']:.2f}) {res['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
