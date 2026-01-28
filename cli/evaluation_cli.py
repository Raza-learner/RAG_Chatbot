import argparse
import json
import os
from pathlib import Path
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

from google import genai
from google.genai import types
import json
from dotenv import load_dotenv
import os

def llm_evaluate_results(query: str, results: list[dict]) -> list[int]:
    """Use Gemini to score each result for relevance (0-3) with structured JSON."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY missing → skipping evaluation")
        return [0] * len(results)

    # 1. Use the new Client pattern
    client = genai.Client(api_key=api_key)

    # 2. Format results for the prompt
    formatted_results = [
        f"ID {i}: {res['title']} - {res['document']}" 
        for i, res in enumerate(results)
    ]
    formatted_str = "\n".join(formatted_results)

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:
    Query: "{query}"

    Results:
    {formatted_str}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Return a JSON list of integers representing the scores in the exact order provided."""

    try:
        # 3. Use GenerateContentConfig to force JSON output
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                # You can also provide a response_schema for 100% reliability
                response_schema={
                    "type": "array",
                    "items": {"type": "integer"}
                }
            )
        )

        # 4. Extract and parse. The SDK's response.text is already cleaned.
        scores = json.loads(response.text)
        
        if len(scores) != len(results):
            print("Score count mismatch → padding with zeros")
            return (scores + [0] * len(results))[:len(results)]
            
        return [int(s) for s in scores]

    except Exception as e:
        print(f"LLM evaluation failed: {e} → using default scores")
        return [0] * len(results)
def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for Precision@k)",
    )

    args = parser.parse_args()
    k = args.limit

    # Load golden dataset
    golden_path = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"
    if not golden_path.exists():
        print(f"Error: golden_dataset.json not found at {golden_path}")
        return

    try:
        with open(golden_path, "r") as f:
            golden_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return

    # Handle both possible formats
    if isinstance(golden_data, list):
        test_cases = golden_data[0].get("test_cases", []) if golden_data else []
    elif isinstance(golden_data, dict):
        test_cases = golden_data.get("test_cases", [])
    else:
        print("Error: golden_dataset.json must be a list or dict")
        return

    if not isinstance(test_cases, list):
        print("Error: 'test_cases' must be a list")
        return

    print(f"Loaded {len(test_cases)} test cases")
    print(f"Loaded {len(test_cases)} test cases from golden_dataset.json")

    documents = load_movies()
    hybrid = HybridSearch(documents)

    print(f"k={k}\n")

    for test_case in test_cases:
        query = test_case.get("query")
        if not query:
            print("Skipping test case with missing or empty query")
            continue

        # Use your actual key: "relevant_docs"
        relevant_titles = set(test_case.get("relevant_docs", []))

        if not relevant_titles:
            print(f"Skipping query '{query}' - no relevant docs")
            continue

        # Run RRF search
        results = hybrid.rrf_search(query, k=60, limit=k)

        retrieved_titles = [res["title"] for res in results]

        relevant_retrieved = len(relevant_titles & set(retrieved_titles))
        precision = relevant_retrieved / k if k > 0 else 0.0
        recall = relevant_retrieved / len(relevant_titles) if relevant_titles else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {query}")
        print(f"  - Precision@{k}: {precision:.4f}")
        print(f"  - Recall@{k}: {recall:.4f}")
        print(f"  - F1 Score: {f1:4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(sorted(relevant_titles))}")
        print()
if __name__ == "__main__":
    main()
