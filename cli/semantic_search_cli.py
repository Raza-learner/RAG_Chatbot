#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
        verify_model, 
        embed_text, 
        verify_embeddings,
        embed_query_text,
        SemanticSearch,
        load_movies,
        chunk_text,
        semantic_chunking,
        ChunkedSemanticSearch,
    )

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add the 'verify' command
    subparsers.add_parser("verify", help="Verify that the embedding model loads correctly")
    
    #Embedding text
    embed_parser = subparsers.add_parser("embed-text", help="Emebedding text")
    embed_parser.add_argument("text", type=str, help="The text to embed")

    subparsers.add_parser("verify-embeddings",help="Verify that movie embeddings are correctly loaded/created")
    embed_query = subparsers.add_parser("embedquery", help="Emebedding text")
    embed_query.add_argument("query",type=str,help="The Search query")

    search_parser = subparsers.add_parser("search", help="Semantic search for movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    
    chunk_parser =  subparsers.add_parser("chunk",help="Chunking of document")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, 
                             help="Number of words per chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0,
                              help="Number of overlapping words between chunks (default: 0)")
     
    semantic_chunk_parser = subparsers.add_parser(
        "semantic-chunk",
        help="Split text into semantic chunks based on sentences"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk (default: 4)"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences between chunks (default: 0)"
    )

    subparsers.add_parser(
        "embed-chunks",
        help="Generate or load embeddings for all document chunks"
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
        case "embedquery":
            print("Embedding query")
            embed_query_text(args.query)
        case "search":
            print(f"Searching semantically for: {args.query}")
            searcher = SemanticSearch()
            documents = load_movies()
            searcher.load_or_create_embeddings(documents)
            results = searcher.search(args.query, limit=args.limit)
            
            if not results:
                print("No results found.")
            else:
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res['title']} (score: {res['score']})")
                    print(f"   {res['description'][:150]}...")  # first 150 chars
                    print()

        case "chunk":
            print(f"Chunking text with size {args.chunk_size},overlapping {args.overlap}...")
            chunk_text(args.text, chunk_size=args.chunk_size,overlap=args.overlap)
       
        case "semantic-chunk":
            print(f"Semantically chunking with max {args.max_chunk_size} sentences, overlap {args.overlap}...")
            semantic_chunking(
                args.text,
                max_chunk_size=args.max_chunk_size,
                overlap=args.overlap
            )
        case "embed-chunks":
            print("Embedding movie chunks...")
            documents = load_movies()
            searcher = ChunkedSemanticSearch()
            embeddings = searcher.load_or_create_chunk_embeddings(documents)
            total_chunks = len(embeddings) if embeddings is not None else 0
            print(f"Generated/Loaded {total_chunks} chunked embeddings")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
