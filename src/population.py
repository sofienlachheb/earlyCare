# src/population.py
import pandas as pd

def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Age" in out.columns:
        out["AgeGroup"] = pd.cut(
            out["Age"],
            bins=[0,18,25,35,45,55,65,120],
            labels=["<18","18-25","26-35","36-45","46-55","56-65","65+"]
        )
    if "age" in out.columns and "AgeGroup" not in out.columns:
        out["AgeGroup"] = pd.cut(
            out["age"],
            bins=[0,18,25,35,45,55,65,120],
            labels=["<18","18-25","26-35","36-45","46-55","56-65","65+"]
        )
    if "BMI" in out.columns:
        out["BMIGroup"] = pd.cut(
            out["BMI"],
            bins=[0,18.5,25,30,35,100],
            labels=["Under","Normal","Over","Obese-I","Obese-II+"]
        )
    return out

def population_summary(df_with_risk: pd.DataFrame) -> pd.DataFrame:
    dfb = bucketize(df_with_risk)
    group_cols = [c for c in ["AgeGroup", "BMIGroup"] if c in dfb.columns]
    if not group_cols:
        return dfb[["risk"]].describe().T

    return (dfb.groupby(group_cols, dropna=False)
            .agg(count=("risk","size"),
                 avg_risk=("risk","mean"),
                 p90_risk=("risk", lambda s: s.quantile(0.9)))
            .reset_index())
