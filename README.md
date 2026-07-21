# Rental Price Prediction Service

Cross-validated regression pipelines (Ridge vs Random Forest vs Gradient Boosting) on housing data, with the winner served behind a documented REST endpoint with request logging and versioned artifacts.

## Dataset
[Ames Housing](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) or any rental/housing CSV. Save as `data/housing.csv`; edit `TARGET` and `FEATURES` in `train.py` to match your columns.

## Quickstart
```bash
pip install -r requirements.txt
python train.py
uvicorn api:app --reload
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"features": {"GrLivArea": 1500, "OverallQual": 6, "YearBuilt": 1995, "TotRmsAbvGrd": 6, "GarageCars": 2}}'
```

## Results

| Model | RMSE | R² |
|---|---|---|
| Ridge (linear baseline) | $38,756 | 0.761 |
| Random Forest | $32,274 | 0.834 |
| Gradient Boosting | $30,608 | 0.851 |

Gradient Boosting wins, explaining ~85% of price variance with a typical
prediction error of ~$30.6k. The gap over the linear baseline (R² 0.761
→ 0.851) suggests real nonlinear interactions between features — e.g.
overall quality likely has a bigger price effect on larger homes than
smaller ones, which tree-based models capture naturally but Ridge cannot.

## API in action

```bash
$ curl -X POST localhost:8000/predict -d '{"features": {"Gr Liv Area": 2000, "Overall Qual": 7,
"Year Built": 2005, "TotRms AbvGrd": 8, "Garage Cars": 2}}'
{"predicted_price": 248378.71, "model_version": "20260721-163711"}

$ curl -X POST localhost:8000/predict -d '{"features": {"Gr Liv Area": 900, "Overall Qual": 4,
"Year Built": 1950, "TotRms AbvGrd": 5, "Garage Cars": 1}}'
{"predicted_price": 98454.54, "model_version": "20260721-163711"}
```

A modern, larger, higher-quality home prices ~2.5x above a small, older,
lower-quality one — consistent with how these features actually drive
price in the training data.