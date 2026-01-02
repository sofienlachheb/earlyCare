# EarlyCare-X (Predict • Explain • Simulate • Population)

⚠️ **Educational / awareness project only. Not a medical diagnosis tool.**

## What you get
- Train models for **Diabetes** and/or **Heart Disease**
- Explain predictions (Top contributing features via SHAP)
- What-if prevention simulator (find small changes that reduce risk)
- Population dashboard (anonymous, aggregated insights)

## Project structure
```
earlycare_x/
  data/
    diabetes.csv
    heart.csv
  models/
  artifacts/
  src/
  app.py
  requirements.txt
```

## 1) Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Put datasets in `data/`
### Diabetes dataset (Pima)
Expected target column: `Outcome`
Common feature columns:
`Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age`

### Heart dataset
We support either:
- A column named `target` (0/1), **or**
- A column named `num` (0..4) which will be converted to binary `target = (num>0)`

If your heart dataset has different column names (e.g., `age, trestbps, chol, ...`), that's fine
as long as they are numeric. The app will load feature names from the trained model.

## 3) Train
```bash
python src/train.py
```

Models saved to `models/` and feature metadata to `artifacts/`.

## 4) Run the demo app
```bash
streamlit run app.py
```

## Notes
- If you get errors about missing columns, your CSV column names don't match.
  Fix by editing `src/train.py` and ensuring the `target` column names are correct.
