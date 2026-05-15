# benchmark/diagnose.py
import pandas as pd, sys
sys.path.insert(0, '/app')
from src.pipeline.predict_pipeline import PredictPipeline

p = PredictPipeline()
df = p.df

print("Ground‑truth titles in dataset?")
for title in ['Edge of Tomorrow', 'Mission: Impossible - Fallout', 'The Last Samurai']:
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        print(f"MISSING: {title}")
    else:
        row = matches.iloc[0]
        has_cruise = 'cruise' in row['cast'].lower()
        print(f"FOUND: {title} → id={row['movie_id']}, vote_count={row['vote_count']}, cast has cruise? {has_cruise}")

print("\nTop 10 Tom Cruise movies (pipeline):")
results = p.recommend('Tom Cruise movies', top_n=10, use_llm_parse=False, generate_explanations=False)
for r in results:
    print(f"{r['title']} ({r['year']})  id={r['movie_id']}")