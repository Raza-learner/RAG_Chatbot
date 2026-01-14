#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add the 'verify' command
    subparsers.add_parser("verify", help="Verify that the embedding model loads correctly")
    
    #Embedding text
    embed_parser = subparsers.add_parser("embed-text", help="Emebedding text")
    embed_parser.add_argument("text", type=str, help="The text to embed")
    subparsers.add_parser(
        "verify-embeddings",
        help="Verify that movie embeddings are correctly loaded/created"
    )
    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Initiated....")
            verify_model()
        case "embed-text":
            embed_text(args.text)
        case "verify-embeddings":
            print("Verifying movie embeddings...")
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
