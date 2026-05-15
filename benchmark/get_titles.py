#!/usr/bin/env python
"""
benchmark/get_titles.py — flexible query runner for the MovieMind pipeline.

Fetches and prints the top-N recommendations for a given list of queries.
Supports Groq NLU and exports results as JSON for later evaluation.

Usage:
  uv run python benchmark/get_titles.py                             # defaults
  uv run python benchmark/get_titles.py --use-llm                   # with Groq NLU
  uv run python benchmark/get_titles.py --top 5 --output eval.json  # custom top-k + save
  uv run python benchmark/get_titles.py --queries "Inception" "Batman"   # custom queries
"""

import os, sys, json, argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ARTIFACTS_DIR", os.path.join(os.getcwd(), "artifacts"))

from src.pipeline.predict_pipeline import PredictPipeline

# Predefined evaluation query list
DEFAULT_QUERIES = [
    "Christopher Nolan movies",
    "Tom Cruise movies",
    "movies like Dune",
    "rom com movies",
    "horror movies after 2020",
    "suggest awesome heist movies",
    "marvel movies without iron man",
    "Martin Scorsese movies",
    "space vibe movies",
    "superhero movies",
]


def main():
    parser = argparse.ArgumentParser(description="MovieMind quick‑query runner")
    parser.add_argument("--top", type=int, default=10, help="Number of results per query")
    parser.add_argument("--use-llm", action="store_true", help="Enable structured Groq NLU parsing")
    parser.add_argument("--output", type=str, help="Save results as JSON for evaluation")
    parser.add_argument("--queries", nargs="*", help="Custom query list; overrides defaults")
    args = parser.parse_args()

    queries = args.queries if args.queries else DEFAULT_QUERIES
    pipeline = PredictPipeline()

    all_results = []
    for q in queries:
        print(f"\n📽️  Query: {q}")
        try:
            results = pipeline.recommend(
                q,
                top_n=args.top,
                use_llm_parse=args.use_llm,
                generate_explanations=False,
            )
            for i, r in enumerate(results):
                print(f"  {i+1:2d}. {r['title']} ({r['year']})")
            all_results.append({"query": q, "results": results})
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results.append({"query": q, "error": str(e)})

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved results for {len(all_results)} queries -> {args.output}")


if __name__ == "__main__":
    main()