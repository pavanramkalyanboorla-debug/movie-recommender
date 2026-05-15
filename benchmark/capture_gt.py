# benchmark/capture_gt.py
import sys, json
sys.path.insert(0, '/app')
from src.pipeline.predict_pipeline import PredictPipeline

pipeline = PredictPipeline()

queries = [
    "Christopher Nolan movies",
    "Tom Cruise movies",
    "movies like Dune",
    "rom com movies",
    "horror movies after 2020"
]

output = []
for q in queries:
    results = pipeline.recommend(q, top_n=10, use_llm_parse=False, generate_explanations=False)
    ids = [r['movie_id'] for r in results]
    titles = [f"{r['title']} ({r['year']})" for r in results]
    print(f"\n{q}")
    for t, i in zip(titles, ids):
        print(f"  {t}  (id={i})")
    output.append({"query": q, "relevant": ids})

with open("benchmark/test_queries_ids.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to benchmark/test_queries_ids.json")