# app.py
import pandas as pd
import streamlit as st
from pathlib import Path

from src.explain import explain_instance, load_features
from src.simulate import load_model, what_if_grid, risk_score
from src.population import population_summary

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="EarlyCare-X", layout="wide")
st.title("EarlyCare-X 🩺 | Predict • Explain • Simulate • Population")
st.caption("⚠️ للتوعية ودعم القرار الوقائي فقط — ليس تشخيصًا طبيًا.")

task = st.sidebar.selectbox("اختر المهمة", ["سكري (Diabetes)", "قلب (Heart)"])

if task.startswith("سكري"):
    model_file = "diab_model.pkl"
    dataset_file = "diabetes.csv"
    target_name = "Outcome"
    # Default editable features for prevention simulation (only if present)
    preferred_editable = ["Glucose", "BMI", "BloodPressure"]
    defaults = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 75,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33
    }
else:
    model_file = "heart_model.pkl"
    dataset_file = "heart.csv"
    # heart target is normalized to "target" during training
    target_name = "target"
    preferred_editable = ["trestbps", "chol"]
    defaults = {"age": 45, "trestbps": 130, "chol": 220}

st.sidebar.write("**خطوات التشغيل**")
st.sidebar.code("1) python src/train.py\n2) streamlit run app.py")

# Load trained model + features
try:
    model = load_model(model_file)
    features = load_features(model_file)
except Exception as e:
    st.error(f"لم يتم العثور على النموذج أو ملفات الخصائص. شغّل التدريب أولاً: python src/train.py\n\nتفاصيل: {e}")
    st.stop()

st.subheader("1) إدخال بيانات الحالة (افتراضية)")
cols = st.columns(4)
inputs = {}
for i, feat in enumerate(features):
    with cols[i % 4]:
        val = defaults.get(feat, 0.0)
        if isinstance(val, int):
            inputs[feat] = st.number_input(feat, value=int(val))
        else:
            inputs[feat] = st.number_input(feat, value=float(val))

x = pd.DataFrame([inputs], columns=features)
base_risk = risk_score(model, x)
st.metric("Risk Score", f"{base_risk:.1f} / 100")

st.subheader("2) تفسير القرار (Top عوامل)")
top = explain_instance(model_file, x, top_k=8)
st.dataframe(pd.DataFrame(top, columns=["Feature", "SHAP contribution"]))

st.subheader("3) محاكاة وقائية (What-If) — أقل تغيير يخفض الخطر")
editable = [f for f in preferred_editable if f in features]
if not editable:
    st.info("لا توجد أعمدة قابلة للمحاكاة من القائمة الافتراضية. يمكنك تعديل preferred_editable في app.py لتناسب أعمدة بياناتك.")
else:
    with st.expander("إعدادات المحاكاة"):
        steps = st.slider("عدد السيناريوهات التي نعرضها", 5, 20, 10)
        action_dict = {}
        for f in editable:
            delta = st.select_slider(
                f"تغييرات {f}",
                options=[-30,-20,-15,-10,-5,-2,0,2,5,10,15,20,30],
                value=0
            )
            action_dict[f] = [delta-5, delta, delta+5] if delta != 0 else [-5, 0, 5]

    best = what_if_grid(model, x, action_dict, top_n=steps)
    rows = []
    for changes, r in best:
        row = {"Risk_after": r, **{f"Δ{k}": v for k, v in changes.items()}}
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).sort_values("Risk_after"))

st.subheader("4) لوحة مجتمع (Population) — بيانات مجهولة ومجمعة")
st.caption("تجميع إحصائي بدون هويات، لتوضيح كيف يمكن توجيه الوقاية على مستوى المجتمع.")

csv_path = DATA / dataset_file
if not csv_path.exists():
    st.warning(f"الملف غير موجود: {csv_path}. ضع dataset في مجلد data/")
else:
    df = pd.read_csv(csv_path)
    if target_name in df.columns:
        dfX = df.drop(columns=[target_name])
    else:
        dfX = df.copy()

    dfX = dfX.reindex(columns=features).fillna(0)
    proba = model.predict_proba(dfX)[:, 1]
    df_pop = dfX.copy()
    df_pop["risk"] = (proba * 100.0)

    summary = population_summary(df_pop)
    st.dataframe(summary)
