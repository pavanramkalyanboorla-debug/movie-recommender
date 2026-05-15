# scripts/build_streaming_lookup.py
import pandas as pd, json, os

CSV_PATH = "data/MoviesOnStreamingPlatforms.csv"   # rename if needed
OUTPUT   = "artifacts/streaming_lookup.json"

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
platforms = [c for c in df.columns if c in ["Netflix", "Hulu", "Prime Video", "Disney+"]]

lookup = {}
for _, row in df.iterrows():
    title = str(row.get("Title", "")).strip()
    year = row.get("Year", None)
    if pd.isna(year) or title == "":
        continue
    year = int(year)
    key = f"{title.lower()}||{year}"
    active = [p for p in platforms if row.get(p, 0) == 1]
    if active:
        lookup[key] = active

os.makedirs("artifacts", exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(lookup, f, indent=2)
print(f"✅ Saved {len(lookup)} entries to {OUTPUT}")