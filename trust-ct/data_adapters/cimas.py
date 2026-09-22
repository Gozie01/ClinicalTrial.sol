"""
Raw Cimas HTN claims loader.

Reads HTN Adherence Data.csv and HTN_Additional Columns.csv exactly as
supplied, applies no imputation, no outcome logic, no feature engineering.
Returns a clean claims DataFrame and a plan-info lookup DataFrame.
"""

import pathlib
import pandas as pd


# Columns that must be dropped before any downstream processing.
# They contain full-year or post-landmark information that would leak outcome.
_LEAKAGE_COLS = ["ADHERENCE", "PZTIQNT NZXQ", "PAYER NAME"]

_DATE_COLS = ["SERVICE DATE", "ASSESS DATE", "DATE RECEIVED"]
_MONEY_COLS = [
    "AMOUNT CLAIMED", "PAID FROM RISK AMT", "PAID FROM THRESHHOLD",
    "PAID FROM SAVINGS", "RECOVERY AMOUNT", "TOTAL AMOUNT PAID",
    "TARIFF", "CO-PAY",
]


def load_claims(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Load raw claims CSV into a clean DataFrame.

    Applies:
    - UTF-8 / latin-1 encoding fallback
    - Strip column whitespace
    - Drop leakage columns
    - Parse SERVICE DATE as datetime (M/D/YYYY)
    - Create PID = str(MEMBER)+"|"+str(INO)
    - Cast numeric columns

    Does NOT apply:
    - Date-window filtering (caller's responsibility)
    - Outcome computation
    - Feature engineering
    """
    path = pathlib.Path(csv_path)
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)

    df.columns = [c.strip() for c in df.columns]

    # Drop leakage columns that exist in this file
    drop = [c for c in _LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop)

    # PID
    df["MEMBER"] = df["MEMBER"].str.strip()
    df["INO"] = df["INO"].str.strip()
    df["pid"] = df["MEMBER"] + "|" + df["INO"]

    # SERVICE DATE: M/D/YYYY in the raw file
    df["SERVICE DATE"] = pd.to_datetime(
        df["SERVICE DATE"].str.strip(), format="%m/%d/%Y", errors="coerce"
    )

    # Numeric casts (money, units, age)
    for col in _MONEY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["UNITS", "CURRENT AGE", "DIS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip leading numeric code from PROVIDER (e.g. "96827 CIMAS FOURTH STREET PHARMACY")
    if "PROVIDER" in df.columns:
        df["provider_name"] = (
            df["PROVIDER"].str.strip().str.replace(r"^\d+\s+", "", regex=True).str.strip()
        )

    # GENDER normalise
    if "GENDER" in df.columns:
        df["sex_female"] = (df["GENDER"].str.upper().str.strip() == "F").astype("Int8")

    return df


def load_plan_info(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Load HTN_Additional Columns.csv.

    Returns DataFrame with columns:
        OPTION NAME, SCHEMETYPE, COVERTYPE, ANNUALCONTRIBUTION
    plus derived ordinal columns scheme_type_ord, cover_type_bin.
    """
    path = pathlib.Path(csv_path)
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin-1")

    df.columns = [c.strip() for c in df.columns]
    df["OPTION NAME"] = df["OPTION NAME"].str.strip().str.upper()
    df["SCHEMETYPE"] = df["SCHEMETYPE"].str.strip().str.upper()
    df["COVERTYPE"] = df["COVERTYPE"].str.strip().str.upper()
    df["ANNUALCONTRIBUTION"] = pd.to_numeric(df["ANNUALCONTRIBUTION"], errors="coerce")

    scheme_ord = {"BASIC": 0, "MEDIUM": 1, "PREMIUM": 2}
    df["scheme_type_ord"] = df["SCHEMETYPE"].map(scheme_ord)
    df["cover_type_bin"] = (df["COVERTYPE"] == "COMPREHENSIVE").astype("Int8")

    return df
