# src/train.py
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
MODELS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

def train_binary_xgb(df: pd.DataFrame, target: str, drop_cols=None, model_name="model.pkl"):
    drop_cols = drop_cols or []
    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ))
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"=== {model_name} ===")
    print("AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred))

    joblib.dump(pipe, MODELS / model_name)

    feature_info = {"features": list(X.columns), "target": target}
    with open(ARTIFACTS / f"{model_name}.features.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

def main():
    diab_path = DATA / "diabetes.csv"
    heart_path = DATA / "heart.csv"

    if diab_path.exists():
        diab = pd.read_csv(diab_path)
        if "Outcome" not in diab.columns:
            raise ValueError("diabetes.csv must contain target column named 'Outcome'.")
        train_binary_xgb(diab, target="Outcome", model_name="diab_model.pkl")
    else:
        print("Skipping diabetes: data/diabetes.csv not found")

    if heart_path.exists():
        heart = pd.read_csv(heart_path)

        if "target" in heart.columns:
            train_binary_xgb(heart, target="target", model_name="heart_model.pkl")
        elif "num" in heart.columns:
            heart["target"] = (heart["num"] > 0).astype(int)
            train_binary_xgb(heart, target="target", drop_cols=["num"], model_name="heart_model.pkl")
        else:
            raise ValueError("heart.csv must contain 'target' (0/1) or 'num' (0..4) column.")
    else:
        print("Skipping heart: data/heart.csv not found")

if __name__ == "__main__":
    main()
