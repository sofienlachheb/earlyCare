# src/explain.py
import json
import joblib
import pandas as pd
from pathlib import Path
import shap

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

def load_features(model_file: str):
    meta_path = ARTIFACTS / f"{model_file}.features.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)["features"]

def explain_instance(model_file: str, x_row: pd.DataFrame, top_k: int = 8):
    """
    Returns list of (feature, shap_contribution) sorted by absolute contribution desc.
    """
    model = joblib.load(MODELS / model_file)

    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(x_row)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)

    contrib = shap_values[0]
    features = x_row.columns
    pairs = list(zip(features, contrib))
    pairs_sorted = sorted(pairs, key=lambda p: abs(p[1]), reverse=True)
    return pairs_sorted[:top_k]
