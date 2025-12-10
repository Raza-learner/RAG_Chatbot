#!/usr/bin/env python3

import argparse
import sys

from lib.inverted_index import InvertedIndex
from lib.search_utils import load_movies


def build_command() -> None:
    print("Building inverted index...")
    movies = load_movies()
    index = InvertedIndex()
    index.build(movies)
    index.save()


def idf_command(term: str) -> None:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'uv run cli/keyword_search_cli.py build' first.")
        sys.exit(1)

    idf_value = index.idf(term)
    print(f"Inverse document frequency of '{term}': {idf_value:.2f}")


def search_command(query: str) -> None:
    print(f"Searching for: {query}")
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'build' first.")
        sys.exit(1)

    results = index.search(query, max_results=5)
    if not results:
        print("No results found.")
    else:
        print(f"\nTop {len(results)} results:")
        for i, (doc_id, title) in enumerate(results, 1):
            print(f"{i}. [ID: {doc_id}] {title}")   # ← FIXED LINE


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build and save the inverted index")

    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for a term")
    idf_parser.add_argument("term", type=str, help="The term to calculate IDF for")

    search_parser = subparsers.add_parser("search", help="Search using the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    if args.command == "build":
        build_command()
    elif args.command == "idf":
        idf_command(args.term)
    elif args.command == "search":
        search_command(args.query)


if __name__ == "__main__":
    main()
