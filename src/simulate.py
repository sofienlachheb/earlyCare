# src/simulate.py
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

def load_model(model_file: str):
    return joblib.load(MODELS / model_file)

def risk_score(model, x: pd.DataFrame) -> float:
    proba = model.predict_proba(x)[:, 1][0]
    return float(proba * 100.0)

def what_if_grid(model, base_row: pd.DataFrame, actions: dict, top_n: int = 10):
    """
    actions example:
    {
      "BMI": [-2, -1, 0, 1],
      "Glucose": [-15, -10, 0],
      "BloodPressure": [-10, 0]
    }
    Returns best scenarios with lowest risk.
    """
    results = []
    keys = list(actions.keys())

    def rec(i, current):
        if i == len(keys):
            x = base_row.copy()
            for k, delta in current.items():
                x.loc[x.index[0], k] = x.loc[x.index[0], k] + delta
            results.append((current.copy(), risk_score(model, x)))
            return
        k = keys[i]
        for delta in actions[k]:
            current[k] = delta
            rec(i + 1, current)
        current.pop(k, None)

    rec(0, {})
    results_sorted = sorted(results, key=lambda r: r[1])
    return results_sorted[:top_n]
