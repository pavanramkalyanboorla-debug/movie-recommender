from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(title="MovieMind Backend")
pipeline = PredictPipeline()

class RecommendRequest(BaseModel):
    query: str
    top_n: int = 10
    w_sim: float = 0.6
    w_rating: float = 0.2
    w_pop: float = 0.2
    use_llm_parse: bool = True
    generate_explanations: bool = True
    exclude_ids: list = []

@app.post("/recommend")
def recommend(req: RecommendRequest):
    results = pipeline.recommend(
        query=req.query,
        top_n=req.top_n,
        w_sim=req.w_sim,
        w_rating=req.w_rating,
        w_pop=req.w_pop,
        use_llm_parse=req.use_llm_parse,
        generate_explanations=req.generate_explanations,
        exclude_ids=req.exclude_ids,
    )
    return {"results": results}

@app.get("/health")
def health():
    return {"status": "healthy"}