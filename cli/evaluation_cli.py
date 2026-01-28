import argparse
import json
import os
from pathlib import Path
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

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

        relevant_in_top_k = len(relevant_titles & set(retrieved_titles))
        precision = relevant_in_top_k / k if k > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{k}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(sorted(relevant_titles))}")
        print()
if __name__ == "__main__":
    main()
