# benchmark/run_eval.py
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.predict_pipeline import PredictPipeline

def map_titles_to_ids(df, items):
    """Convert a list of titles or IDs to a set of movie_id values."""
    ids = set()
    if not items:
        return ids
    if isinstance(items[0], int):
        # Already IDs – use directly
        return set(items)
    # Otherwise, assume strings (titles)
    df_lower = df['title'].str.lower()
    for t in items:
        matches = df[df_lower == t.lower()]
        if not matches.empty:
            ids.add(int(matches.iloc[0]['movie_id']))
        else:
            print(f"   ⚠️  Title not found: {t}")
    return ids

def load_queries(path):
    with open(path, 'r') as f:
        return json.load(f)

def evaluate(pipeline, queries, k=10, use_llm=False):
    recall_vals, precision_vals = [], []
    df = pipeline.df

    for q_data in queries:
        query = q_data['query']
        print(f"📽️  Evaluating: {query}")
        relevant = map_titles_to_ids(df, q_data.get('relevant', []))
        if not relevant:
            print("   ⚠️  No ground truth – skipping")
            continue
        results = pipeline.recommend(query, top_n=k, use_llm_parse=use_llm, generate_explanations=False)
        retrieved_ids = {r['movie_id'] for r in results}
        hits = len(retrieved_ids & relevant)
        recall = hits / len(relevant)
        precision = hits / k
        recall_vals.append(recall)
        precision_vals.append(precision)
        print(f"   Recall: {recall:.2f} | Precision: {precision:.2f} | GT size: {len(relevant)}")

    if recall_vals:
        print("\n" + "="*60)
        print(f"📊 Average Recall@{k} : {np.mean(recall_vals):.3f}")
        print(f"📊 Average Precision@{k}: {np.mean(precision_vals):.3f}")
    else:
        print("No evaluable queries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', default='benchmark/test_queries.json')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--use_llm', action='store_true')
    args = parser.parse_args()
    pipeline = PredictPipeline()
    queries = load_queries(args.queries)
    print(f"🚀 Evaluating {len(queries)} queries (k={args.top}, use_llm={args.use_llm})…")
    evaluate(pipeline, queries, k=args.top, use_llm=args.use_llm)