#!/usr/bin/env python3

import argparse
import mimetypes
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

def main():
    parser = argparse.ArgumentParser(description="Multimodal Query Rewriting CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file (e.g. data/paddington.jpeg)")
    parser.add_argument("--query", type=str, required=True, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    # Load API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return

    # System prompt
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    client = genai.Client(api_key=api_key)

    # Read image file
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Guess MIME type
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    

    # Build content parts
    content = [
        {"text": system_prompt},
        {"inline_data": {"mime_type": mime_type, "data": img_bytes}},
        {"text": args.query.strip()}
    ]

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content
        )
        rewritten = response.text.strip()

        print(f"Rewritten query: {rewritten}")
        if response.usage_metadata:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    except Exception as e:
        print(f"Error during Gemini call: {e}")

if __name__ == "__main__":
    main()
