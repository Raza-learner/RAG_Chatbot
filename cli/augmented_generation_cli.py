#!/usr/bin/env python3

import argparse
import os
from dotenv import load_dotenv
from google import genai 
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Perform Sumarization"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for Summarize")
    summarize_parser.add_argument("--limit", type=int, default=5,help="Limit Summarize")
   
    question_parser = subparsers.add_parser("question", help="Answer a question using RAG")
    question_parser.add_argument("question", type=str, help="The question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of search results to use (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            # Load movies and initialize hybrid search
            documents = load_movies()
            hybrid = HybridSearch(documents)

            # Step 1: Run RRF search (top 5 results)
            print(f"Searching for: {query}...")
            results = hybrid.rrf_search(query, k=60, limit=5)

            # Step 2: Format documents for prompt
            docs_str = ""
            for i, res in enumerate(results, 1):
                docs_str += f"[{i}] {res['title']}\n   {res['document']}\n\n"

            # Step 3: Prepare RAG prompt
            # Note: I added a instruction to cite the source index [1], [2], etc.
            prompt = f"""You are a helpful assistant for Hoopla, a movie streaming service. 
            Answer the user's query using ONLY the provided movie documents. 
            Cite the documents used by their index number (e.g., [1]).

            Query: {query}

            Documents:
            {docs_str}

            Answer:"""

            # Step 4: Setup Client and Generate
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY not found in .env")
                return

            # Use the new Client pattern (v1 SDK)
            client = genai.Client(api_key=api_key)

            try:
               
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=prompt
                )
                rag_answer = response.text.strip()
            except Exception as e:
                print(f"Gemini API error: {e}")
                rag_answer = "Failed to generate answer."

            # Step 5: Print results
            print("\n--- Top Search Matches ---")
            for res in results:
                print(f"• {res['title']}")

            print(f"\n--- Hoopla Assistant Answer ---\n{rag_answer}\n")

        case "summarize":
            query = args.query
            limit = args.limit

            # Step 1: Load movies and initialize hybrid search
            documents = load_movies()
            hybrid = HybridSearch(documents)

            # Step 2: Run RRF search
            print(f"Searching and summarizing for: {query}")
            results = hybrid.rrf_search(query, k=60, limit=limit)

            # Step 3: Format results with clear boundaries
            results_str = "\n".join([
                f"[{i}] {res['title']}: {res['document']}" 
                for i, res in enumerate(results, 1)
            ])

           
            prompt = f"""
            You are a movie expert for Hoopla, a streaming service. 
            Synthesize a comprehensive overview for the user query: "{query}"
            
            Based on these movies:
            {results_str}

            Instructions:
            1. Provide a information-dense 3-4 sentence summary.
            2. Highlight genres, plot connections, and unique themes.
            3. Mention 2-3 specific titles from the list that best fit the query.
            """

            # Step 5: Unified Client Logic
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY not found.")
                return

            client = genai.Client(api_key=api_key)

            try:
                # Use Gemini 2.0 Flash for low-latency, high-quality summarization
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                summary = response.text.strip()
            except Exception as e:
                print(f"Gemini summarization failed: {e}")
                summary = "Failed to generate summary."

            # Step 6: Print Final Results
            print("\nTop Search Matches:")
            for res in results:
                print(f"  • {res['title']}")
            
            print(f"\nLLM Summary:\n{summary}\n")
        case "question":
            question = args.question
            limit = args.limit

            # Load movies & setup hybrid search
            documents = load_movies()
            hybrid = HybridSearch(documents)

            # Perform RRF search
            print(f"Answering question: {question}")
            results = hybrid.rrf_search(question, k=60, limit=limit)

            # Format context for prompt
            context = ""
            for i, res in enumerate(results, 1):
                context += f"{i}. {res['title']}\n   {res['document']}...\n\n"

            # Prepare prompt for Gemini
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY not found.")
                return

            client = genai.Client(api_key=api_key)

            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                # CHANGE: renamed 'summary' to 'answer' to match your print statement
                answer = response.text.strip() 
            except Exception as e:
                print(f"Gemini generation failed: {e}")
                answer = "Failed to generate answer." # CHANGE: match variable name here too

            # Print in requested format
            print("\nSearch Results:")
            for res in results:
                print(f"  - {res['title']}")

            print(f"\nAnswer:\n{answer}")        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
