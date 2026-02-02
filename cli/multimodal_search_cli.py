#!/usr/bin/env python3

import argparse
from lib.multimodal_search import verify_image_embedding,image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify-image-embedding",
        help="Verify image embedding generation"
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )
    image_search_parser = subparsers.add_parser(
        "image-search",
        help="Search movies using an uploaded image"
    )
    image_search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )
    args = parser.parse_args()

    match args.command:
        case "verify-image-embedding":
            verify_image_embedding(args.image_path)

        case "image-search":
            print(f"Searching movies similar to image: {args.image_path}")
            results = image_search_command(args.image_path)

            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['similarity']:.3f})")
                print(f"   {res['description'][:150]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
