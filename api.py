"""Serve the trained model with request logging."""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(filename="predictions.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")

meta = json.loads(Path("models/latest.json").read_text())
model = joblib.load(f"models/model-{meta['version']}.joblib")
app = FastAPI(title="Rental Price API", version=meta["version"])


class Request(BaseModel):
    features: dict


@app.post("/predict")
def predict(req: Request):
    missing = [f for f in meta["features"] if f not in req.features]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    X = pd.DataFrame([req.features])[meta["features"]]
    price = float(model.predict(X)[0])
    logging.info(json.dumps({"input": req.features, "prediction": price}))
    return {"predicted_price": round(price, 2), "model_version": meta["version"]}
