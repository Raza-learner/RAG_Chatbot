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
    # Removed hardcoded merida test as requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build command
    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")

    # search command
    search_parser = subparsers.add_parser("search", help="Search using the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    if args.command == "build":
        build_command()
        return

    # === SEARCH COMMAND ===
    if args.command == "search":
        query = args.query
        print(f"Searching for: {query}")

        index = InvertedIndex()

        try:
            index.load()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run 'python -m cli keyword_search_cli build' first to create the index.")
            sys.exit(1)
        except Exception as e:
            print(f"Failed to load index: {e}")
            sys.exit(1)

        results = index.search(query, max_results=5)

        if not results:
            print("No results found.")
        else:
            print(f"\nTop {len(results)} results:")
            for i, (doc_id, title) in enumerate(results, 1):
                print(f"{i}. [ID: {doc_id}] {title}")


if __name__ == "__main__":
    main()
