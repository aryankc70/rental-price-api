"""Compare regression models with cross-validation; save the best."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "SalePrice"
FEATURES = ["GrLivArea", "OverallQual", "YearBuilt", "TotRmsAbvGrd", "GarageCars"]

df = pd.read_csv("data/housing.csv")
X, y = df[FEATURES], df[TARGET]

pre = ColumnTransformer([("num", Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
]), FEATURES)])

candidates = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
}

best_name, best_score, best_pipe = None, -np.inf, None
for name, est in candidates.items():
    pipe = Pipeline([("pre", pre), ("model", est)])
    cv = cross_validate(pipe, X, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"))
    rmse = -cv["test_neg_root_mean_squared_error"].mean()
    r2 = cv["test_r2"].mean()
    print(f"{name:18s} RMSE={rmse:,.0f}  R2={r2:.3f}")
    if r2 > best_score:
        best_name, best_score, best_pipe = name, r2, pipe

best_pipe.fit(X, y)
Path("models").mkdir(exist_ok=True)
version = time.strftime("%Y%m%d-%H%M%S")
joblib.dump(best_pipe, f"models/model-{version}.joblib")
Path("models/latest.json").write_text(json.dumps(
    {"version": version, "model": best_name, "features": FEATURES}))
print(f"\nSaved {best_name} as models/model-{version}.joblib")
