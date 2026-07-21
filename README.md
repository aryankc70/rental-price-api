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
| Model | CV RMSE | CV R² |
|---|---|---|
| Ridge | _fill in_ | _fill in_ |
| Random Forest | _fill in_ | _fill in_ |
| Gradient Boosting | _fill in_ | _fill in_ |
