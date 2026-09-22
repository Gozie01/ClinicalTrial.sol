import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EICU_DIR = ROOT / "MIMIC Dataset" / "eicu-collaborative-research-database-demo-2.0.1"
CACHE_DIR = ROOT / "evaluation" / "prepared_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(path: Path, usecols=None):
    return pd.read_csv(path, compression="gzip", usecols=usecols, low_memory=False)


def _normalize_age(series: pd.Series):
    age = series.astype(str).str.replace(">", "", regex=False)
    age = pd.to_numeric(age, errors="coerce")
    age = age.fillna(age.median())
    return age


def _prepare_patient_table():
    patient = _safe_read_csv(
        EICU_DIR / "patient.csv.gz",
        usecols=[
            "patientunitstayid",
            "hospitalid",
            "gender",
            "age",
            "ethnicity",
            "admissionheight",
            "admissionweight",
            "hospitaldischargestatus",
            "unitdischargestatus",
            "unitdischargeoffset",
            "unittype",
            "apacheadmissiondx",
        ],
    )
    patient["age"] = _normalize_age(patient["age"])
    patient["admissionheight"] = pd.to_numeric(patient["admissionheight"], errors="coerce")
    patient["admissionweight"] = pd.to_numeric(patient["admissionweight"], errors="coerce")
    patient["unitdischargeoffset"] = pd.to_numeric(patient["unitdischargeoffset"], errors="coerce")
    patient["gender"] = patient["gender"].fillna("Unknown")
    patient["ethnicity"] = patient["ethnicity"].fillna("Unknown")
    patient["unittype"] = patient["unittype"].fillna("Unknown")
    patient["apacheadmissiondx"] = patient["apacheadmissiondx"].fillna("Unknown")

    outcome = (
        patient["unitdischargestatus"].fillna(patient["hospitaldischargestatus"]).astype(str).str.lower().eq("expired")
    ).astype(np.int32)
    patient["Outcome"] = outcome
    return patient


def _prepare_aps_table():
    aps = _safe_read_csv(EICU_DIR / "apacheApsVar.csv.gz")
    aps = aps.drop(columns=["apacheapsvarid"], errors="ignore")
    numeric_cols = [col for col in aps.columns if col != "patientunitstayid"]
    for col in numeric_cols:
        aps[col] = pd.to_numeric(aps[col], errors="coerce").replace(-1, np.nan)
    aps = aps.groupby("patientunitstayid", as_index=False).mean(numeric_only=True)
    aps = aps.add_prefix("aps_")
    aps = aps.rename(columns={"aps_patientunitstayid": "patientunitstayid"})
    return aps


def _prepare_apache_pred_table():
    pred = _safe_read_csv(EICU_DIR / "apachePredVar.csv.gz")
    pred = pred.drop(columns=["apachepredvarid"], errors="ignore")
    keep_cols = [
        "patientunitstayid",
        "sicuday",
        "saps3day1",
        "gender",
        "teachtype",
        "region",
        "bedcount",
        "graftcount",
        "meds",
        "verbal",
        "motor",
        "eyes",
        "age",
        "thrombolytics",
        "aids",
        "hepaticfailure",
        "lymphoma",
        "metastaticcancer",
        "leukemia",
        "immunosuppression",
        "cirrhosis",
        "electivesurgery",
        "activetx",
        "readmit",
        "ventday1",
        "diabetes",
        "creatinine",
    ]
    pred = pred[[col for col in keep_cols if col in pred.columns]].copy()
    for col in pred.columns:
        if col == "patientunitstayid":
            continue
        if pred[col].dtype == object:
            pred[col] = pd.Categorical(pred[col]).codes
        pred[col] = pd.to_numeric(pred[col], errors="coerce").replace(-1, np.nan)
    pred = pred.groupby("patientunitstayid", as_index=False).mean(numeric_only=True)
    pred = pred.add_prefix("pred_")
    pred = pred.rename(columns={"pred_patientunitstayid": "patientunitstayid"})
    return pred


def _prepare_lab_table():
    labs = _safe_read_csv(
        EICU_DIR / "lab.csv.gz",
        usecols=["patientunitstayid", "labname", "labresult"],
    )
    labs["labresult"] = pd.to_numeric(labs["labresult"], errors="coerce")
    top_labs = [
        "glucose",
        "creatinine",
        "BUN",
        "sodium",
        "potassium",
        "HCO3",
        "WBC x 1000",
        "Hgb",
        "platelets x 1000",
    ]
    normalized = {lab.lower(): lab.lower().replace(" ", "_").replace("/", "_") for lab in top_labs}
    labs["labname_norm"] = labs["labname"].astype(str).str.lower()
    labs = labs[labs["labname_norm"].isin(normalized.keys())].copy()
    if labs.empty:
        return pd.DataFrame(columns=["patientunitstayid"])
    labs["labname_norm"] = labs["labname_norm"].map(normalized)
    pivot = (
        labs.pivot_table(
            index="patientunitstayid",
            columns="labname_norm",
            values="labresult",
            aggfunc=["mean", "min", "max"],
        )
        .sort_index(axis=1)
    )
    pivot.columns = [f"lab_{agg}_{name}" for agg, name in pivot.columns]
    return pivot.reset_index()


def _prepare_vitals_table():
    vitals = _safe_read_csv(
        EICU_DIR / "vitalPeriodic.csv.gz",
        usecols=[
            "patientunitstayid",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ],
    )
    vital_cols = [col for col in vitals.columns if col != "patientunitstayid"]
    for col in vital_cols:
        vitals[col] = pd.to_numeric(vitals[col], errors="coerce")
    grouped = vitals.groupby("patientunitstayid")[vital_cols].agg(["mean", "min", "max", "std"])
    grouped.columns = [f"vital_{col}_{agg}" for col, agg in grouped.columns]
    return grouped.reset_index()


def _group_hospitals(patient_df: pd.DataFrame, n_groups: int = 8):
    counts = patient_df.groupby("hospitalid").size().sort_values(ascending=False)
    hospital_ids = counts.index.tolist()
    group_loads = [0] * n_groups
    assignment = {}
    for hospital_id in hospital_ids:
        group_idx = int(np.argmin(group_loads))
        assignment[hospital_id] = group_idx
        group_loads[group_idx] += int(counts[hospital_id])
    return patient_df["hospitalid"].map(assignment).astype(np.int32)


def prepare_eicu_demo_dataset(cache_name: str = "eicu_demo_prepared.csv", n_site_groups: int = 8, refresh: bool = False):
    cache_path = CACHE_DIR / cache_name
    meta_path = CACHE_DIR / cache_name.replace(".csv", ".json")
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
        return df, cache_path, meta_path

    patient = _prepare_patient_table()
    aps = _prepare_aps_table()
    pred = _prepare_apache_pred_table()
    lab = _prepare_lab_table()
    vitals = _prepare_vitals_table()

    df = patient.merge(aps, on="patientunitstayid", how="left")
    df = df.merge(pred, on="patientunitstayid", how="left")
    df = df.merge(lab, on="patientunitstayid", how="left")
    df = df.merge(vitals, on="patientunitstayid", how="left")

    df["site_group"] = _group_hospitals(df, n_groups=n_site_groups)

    categorical_cols = ["gender", "ethnicity", "unittype", "apacheadmissiondx"]
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)

    numeric_cols = [col for col in df.columns if col not in {"patientunitstayid", "hospitalid", "Outcome", "site_group"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    medians = df[numeric_cols].median(numeric_only=True)
    df[numeric_cols] = df[numeric_cols].fillna(medians).fillna(0.0)

    df.to_csv(cache_path, index=False)
    meta = {
        "rows": int(len(df)),
        "positive_outcomes": int(df["Outcome"].sum()),
        "n_hospitals": int(df["hospitalid"].nunique()),
        "n_site_groups": int(df["site_group"].nunique()),
        "n_features": int(len([col for col in df.columns if col not in {"patientunitstayid", "hospitalid", "Outcome", "site_group"}])),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return df, cache_path, meta_path
