#!/usr/bin/env python3

import argparse
from lib.multimodal_search import verify_image_embedding

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

    args = parser.parse_args()

    match args.command:
        case "verify-image-embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
