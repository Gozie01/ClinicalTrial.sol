"""
TRUST-CT Phase 2 — Cimas HTN Landmark Cohort Builder

Landmark formulation: P(Y_adh=1 | features available by 2022-06-30)
  - Observation window: January–June 2022 (features computed here)
  - Outcome window:     July–December 2022 (Y_adh computed here)
  - Decision point:     2022-06-30

Y_adh = 1  iff  patient has >=5 months with >=1 antihypertensive refill
               claim in July–December 2022.

NOT confirmed medication ingestion (no bottle-open data).
NOT conventional PDC (no days-supply or prescribed daily dose).

Run:
    cd "C:/Users/Gozie/Desktop/Blockchain in Clinical Trial"
    python trust-ct/cohort/cimas_landmark.py

Outputs (in trust-ct/processed/cimas/):
    cimas_htn_claims_clean.parquet    — obs-window claims, leakage cols removed
    cimas_htn_landmark_6m.parquet     — one row per patient: features + Y_adh + client
    cimas_provider_manifest.csv       — provider-level stats and client assignments
    cimas_cohort_flow.json            — N at each exclusion step + 9 automated checks
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — all relative to the repo root (one level above trust-ct/)
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent          # trust-ct/cohort/
_REPO = _HERE.parent                                      # trust-ct/
_ROOT = _REPO.parent                                      # workspace root

RAW_CSV = (
    _ROOT
    / "Fedlearn"
    / "New"
    / "Datasets- Diabetes and Hypertension Data"
    / "Datasets- Diabetes and Hypertension Data"
    / "Datasets Before Wrangling"
    / "HTN Adherence Data.csv"
)
ADDL_CSV = (
    _ROOT
    / "Fedlearn"
    / "New"
    / "Datasets- Diabetes and Hypertension Data"
    / "Datasets- Diabetes and Hypertension Data"
    / "Additional Columns"
    / "HTN_Additional Columns.csv"
)
OUT_DIR = _REPO / "processed" / "cimas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Protocol constants (match configs/cimas_landmark.yaml — DO NOT change)
# ---------------------------------------------------------------------------
OBS_START = pd.Timestamp("2022-01-01")
OBS_END   = pd.Timestamp("2022-06-30")
OUT_START = pd.Timestamp("2022-07-01")
OUT_END   = pd.Timestamp("2022-12-31")

OUTCOME_MONTH_THRESHOLD = 5   # >= 5 refill months in out window → Y_adh=1
MIN_AGE = 16
PRIMARY_CLIENT_THRESHOLD   = 100   # providers with >= N unique PIDs in obs window
SENSITIVITY_CLIENT_THRESHOLD = 50

SEEDS = [7, 11, 19, 23, 37]

# Columns that must NEVER appear in the feature matrix
_LEAKAGE_COLS = {
    "ADHERENCE",      # full-year monthly adherence count (outcome proxy)
    "PZTIQNT NZXQ",   # obfuscated patient name (identifier)
    "PAYER NAME",     # beneficiary identifier
}

# ---------------------------------------------------------------------------
# Step 1 — Load raw claims
# ---------------------------------------------------------------------------

def _load_raw(csv_path: pathlib.Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_plan(csv_path: pathlib.Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df["OPTION NAME"] = df["OPTION NAME"].str.strip().str.upper()
    df["SCHEMETYPE"]  = df["SCHEMETYPE"].str.strip().str.upper()
    df["COVERTYPE"]   = df["COVERTYPE"].str.strip().str.upper()
    df["ANNUALCONTRIBUTION"] = pd.to_numeric(df["ANNUALCONTRIBUTION"], errors="coerce")
    scheme_map = {"BASIC": 0, "MEDIUM": 1, "PREMIUM": 2}
    df["scheme_type_ord"] = df["SCHEMETYPE"].map(scheme_map)
    df["cover_type_bin"]  = (df["COVERTYPE"] == "COMPREHENSIVE").astype("Int8")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Clean, parse, create PID
# ---------------------------------------------------------------------------

def build_clean_claims(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Drop leakage columns present in this file
    drop_cols = [c for c in _LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # PID
    df["pid"] = df["MEMBER"].str.strip() + "|" + df["INO"].str.strip()

    # SERVICE DATE: raw format D/M/YYYY (e.g. "1/4/2022" = Apr 1).
    # Confirmed by BIRTHDATE="16/12/1996" where day=16 > 12, ruling out M/D/YYYY.
    df["service_date"] = pd.to_datetime(
        df["SERVICE DATE"].str.strip(), format="%d/%m/%Y", errors="coerce"
    )

    # Restrict to 2022 (primary analysis)
    df = df[df["service_date"].dt.year == 2022].copy()

    # Numeric casts
    for col in ["UNITS", "CURRENT AGE", "DIS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["TOTAL AMOUNT PAID", "CO-PAY", "AMOUNT CLAIMED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise GENDER → sex_female
    if "GENDER" in df.columns:
        df["sex_female"] = (
            df["GENDER"].str.strip().str.upper() == "F"
        ).astype("Int8")

    # Strip leading numeric code from PROVIDER
    if "PROVIDER" in df.columns:
        df["provider_name"] = (
            df["PROVIDER"].str.strip()
              .str.replace(r"^\d+\s+", "", regex=True)
              .str.strip()
        )

    # Normalise OPTION NAME for join
    if "OPTION NAME" in df.columns:
        df["OPTION NAME"] = df["OPTION NAME"].str.strip().str.upper()

    return df


# ---------------------------------------------------------------------------
# Step 3 — Apply cohort inclusion filters
# ---------------------------------------------------------------------------

def apply_inclusion(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    flow = {}
    flow["raw_2022_claims"] = len(df)
    flow["raw_2022_pids"]   = df["pid"].nunique()

    # Age >= 16 (use CURRENT AGE; keep row if CURRENT AGE missing to not over-exclude)
    if "CURRENT AGE" in df.columns:
        too_young = df["CURRENT AGE"].notna() & (df["CURRENT AGE"] < MIN_AGE)
        df = df[~too_young].copy()
    flow["after_age_filter_claims"] = len(df)
    flow["after_age_filter_pids"]   = df["pid"].nunique()

    # Restrict to obs window for feature computation
    obs = df[df["service_date"].between(OBS_START, OBS_END)].copy()
    flow["obs_window_claims"] = len(obs)
    flow["obs_window_pids"]   = obs["pid"].nunique()

    return df, obs, flow


# ---------------------------------------------------------------------------
# Step 4 — Compute outcome Y_adh from out window
# ---------------------------------------------------------------------------

def compute_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pid, count distinct calendar months in Jul-Dec 2022 with >=1 claim.
    Y_adh = 1 if count >= 5.
    PIDs with no out-window claims get Y_adh=0.
    """
    out = df[df["service_date"].between(OUT_START, OUT_END)].copy()
    out["out_month"] = out["service_date"].dt.month  # 7-12

    refill_months = (
        out.groupby("pid")["out_month"]
           .nunique()
           .rename("out_refill_months")
           .reset_index()
    )
    refill_months["Y_adh"] = (
        refill_months["out_refill_months"] >= OUTCOME_MONTH_THRESHOLD
    ).astype(int)
    return refill_months


# ---------------------------------------------------------------------------
# Step 5 — Compute landmark features from obs window only
# ---------------------------------------------------------------------------

def compute_features(obs: pd.DataFrame, plan_info: pd.DataFrame) -> pd.DataFrame:
    """
    All features use ONLY obs-window claims (SERVICE DATE in Jan–Jun 2022).
    No Jul-Dec dates, no ADHERENCE column, no full-year totals.
    """
    obs = obs.copy()
    obs["obs_month"] = obs["service_date"].dt.month   # 1-6
    landmark_date    = OBS_END                         # 2022-06-30

    records = []
    for pid, grp in obs.groupby("pid"):
        r = {"pid": pid}

        # --- Demographic (from first record; fairly stable within a year) ---
        r["age"]        = grp["CURRENT AGE"].dropna().iloc[0] if grp["CURRENT AGE"].notna().any() else np.nan
        r["sex_female"] = grp["sex_female"].dropna().iloc[0]   if grp["sex_female"].notna().any()  else np.nan

        # --- Insurance plan (from OPTION NAME join; constant per PID) ---
        opt = grp["OPTION NAME"].dropna().iloc[0] if "OPTION NAME" in grp and grp["OPTION NAME"].notna().any() else None
        r["option_name"] = opt

        # --- Refill-pattern features (critical predictors) ---
        refill_months_set = set(grp["obs_month"].dropna().astype(int).tolist())
        r["n_refill_months_obs"] = len(refill_months_set)

        last_date = grp["service_date"].max()
        r["refill_recency_days"] = (landmark_date - last_date).days

        # Early (Jan-Mar) vs late (Apr-Jun) adherence within obs window
        r["early_months"] = len(refill_months_set & {1, 2, 3})
        r["late_months"]  = len(refill_months_set & {4, 5, 6})

        # Gap detection: any 2-month gap between consecutive refill months
        sorted_months = sorted(refill_months_set)
        has_gap = False
        for i in range(len(sorted_months) - 1):
            if sorted_months[i + 1] - sorted_months[i] >= 2:
                has_gap = True
                break
        r["has_2m_gap"] = int(has_gap)

        # Month of first claim in obs window
        r["first_claim_month"] = grp["service_date"].min().month

        # --- Utilisation features ---
        r["n_claims"]       = len(grp)
        r["n_claim_dates"]  = grp["service_date"].nunique()

        # Drug product diversity
        if "CLM CODE" in grp.columns:
            r["n_products"] = grp["CLM CODE"].nunique()
        else:
            r["n_products"] = np.nan

        # Provider diversity (visits to multiple dispensing sites)
        if "provider_name" in grp.columns:
            r["n_providers_vis"] = grp["provider_name"].nunique()
        else:
            r["n_providers_vis"] = np.nan

        # Pharmacy network hops
        if "AS AT NETWORKS" in grp.columns:
            r["n_networks"] = grp["AS AT NETWORKS"].nunique()
        else:
            r["n_networks"] = np.nan

        # Claim amounts and units (obs window only — per-claim, not full-year)
        if "TOTAL AMOUNT PAID" in grp.columns:
            r["obs_amount"] = grp["TOTAL AMOUNT PAID"].sum(skipna=True)
        else:
            r["obs_amount"] = np.nan

        if "UNITS" in grp.columns:
            r["obs_units"] = grp["UNITS"].sum(skipna=True)
        else:
            r["obs_units"] = np.nan

        if "CO-PAY" in grp.columns:
            r["copay_sum"] = grp["CO-PAY"].sum(skipna=True)
        else:
            r["copay_sum"] = np.nan

        # Derived per-claim ratios
        r["amount_per_claim"] = r["obs_amount"] / r["n_claims"] if r["n_claims"] > 0 else np.nan
        r["units_per_claim"]  = r["obs_units"]  / r["n_claims"] if r["n_claims"] > 0 else np.nan

        records.append(r)

    feat = pd.DataFrame(records)

    # Log-transform ANNUALCONTRIBUTION (from plan_info join)
    feat = feat.merge(
        plan_info[["OPTION NAME", "scheme_type_ord", "cover_type_bin", "ANNUALCONTRIBUTION"]],
        left_on="option_name", right_on="OPTION NAME", how="left"
    ).drop(columns=["OPTION NAME"])

    feat["annual_contrib_log"] = np.log1p(feat["ANNUALCONTRIBUTION"])

    # Final leakage guard — verify no leakage column survived
    for col in _LEAKAGE_COLS:
        assert col not in feat.columns, f"Leakage column {col!r} found in features!"

    return feat


# ---------------------------------------------------------------------------
# Step 6 — Client assignment
# ---------------------------------------------------------------------------

def assign_clients(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Each PID's client = first dispensing provider in obs window (by SERVICE DATE).
    Client = provider_name with >= PRIMARY_CLIENT_THRESHOLD unique PIDs.
    PIDs assigned to sub-threshold providers → client="other".
    """
    # First provider by date for each PID
    first_prov = (
        obs.sort_values("service_date")
           .groupby("pid")["provider_name"]
           .first()
           .reset_index()
           .rename(columns={"provider_name": "assigned_provider"})
    )

    # Count unique PIDs per provider
    provider_counts = (
        first_prov["assigned_provider"]
        .value_counts()
        .rename("n_pids")
        .rename_axis("assigned_provider")
        .reset_index()
    )

    # Client tiers
    provider_counts["client_primary"] = (
        provider_counts["n_pids"] >= PRIMARY_CLIENT_THRESHOLD
    )
    provider_counts["client_sensitivity"] = (
        provider_counts["n_pids"] >= SENSITIVITY_CLIENT_THRESHOLD
    )

    first_prov = first_prov.merge(provider_counts, on="assigned_provider", how="left")

    # Assign client label
    first_prov["client"] = first_prov.apply(
        lambda r: r["assigned_provider"] if r["client_primary"] else "other",
        axis=1
    )
    first_prov["client_sens"] = first_prov.apply(
        lambda r: r["assigned_provider"] if r["client_sensitivity"] else "other",
        axis=1
    )

    # Provider manifest
    manifest = provider_counts.copy()
    manifest = manifest.sort_values("n_pids", ascending=False)

    return first_prov[["pid", "assigned_provider", "client", "client_sens"]], manifest


# ---------------------------------------------------------------------------
# Step 7 — 9 Automated checks
# ---------------------------------------------------------------------------

def run_checks(landmark: pd.DataFrame, manifest: pd.DataFrame, flow: dict) -> list[dict]:
    checks = []

    def chk(name: str, passed: bool, detail: str):
        checks.append({"check": name, "passed": passed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    n = len(landmark)
    prev = n

    # C1 — Leakage guard: ADHERENCE absent from feature matrix
    leakage_cols_present = [c for c in _LEAKAGE_COLS if c in landmark.columns]
    chk("C1_leakage_free",
        len(leakage_cols_present) == 0,
        f"Leakage columns in landmark: {leakage_cols_present or 'none'}")

    # C2 — Outcome window dates absent from obs features
    # (structural guarantee from compute_features — verify no out_month column snuck in)
    chk("C2_no_out_window_dates",
        "out_month" not in landmark.columns,
        "No out_month column in landmark feature set")

    # C3 — Cohort size
    chk("C3_cohort_size",
        n >= 10_000,
        f"N={n:,} (need >=10,000)")

    # C4 — Outcome prevalence
    prev_rate = landmark["Y_adh"].mean()
    chk("C4_outcome_prevalence",
        0.25 <= prev_rate <= 0.70,
        f"Y_adh prevalence={prev_rate:.3f} (need 0.25-0.70)")

    # C5 — Y_adh binary and complete
    n_nan = landmark["Y_adh"].isna().sum()
    vals  = set(landmark["Y_adh"].dropna().unique().tolist())
    chk("C5_outcome_complete",
        n_nan == 0 and vals.issubset({0, 1}),
        f"Y_adh NaN={n_nan}, unique values={sorted(vals)}")

    # C6 — Client count (primary)
    n_primary_clients = manifest[manifest["client_primary"]]["assigned_provider"].nunique()
    chk("C6_client_count",
        n_primary_clients >= 10,
        f"Primary clients (>=100 PIDs): {n_primary_clients} (need >=10)")

    # C7 — Client assignment coverage
    n_assigned = (landmark["client"] != "other").sum()
    frac = n_assigned / n if n > 0 else 0
    chk("C7_client_coverage",
        frac >= 0.60,
        f"{n_assigned:,}/{n:,} = {frac:.1%} assigned to named clients (need >=60%)")

    # C8 — PID uniqueness
    n_dupes = landmark.duplicated(subset="pid").sum()
    chk("C8_pid_unique",
        n_dupes == 0,
        f"Duplicate PIDs={n_dupes}")

    # C9 — Feature missing rate: no feature >60% missing
    feat_cols = [c for c in landmark.columns
                 if c not in {"pid", "Y_adh", "out_refill_months",
                               "client", "client_sens", "assigned_provider",
                               "ANNUALCONTRIBUTION"}]
    missing = landmark[feat_cols].isna().mean()
    worst = missing.max() if len(missing) > 0 else 0.0
    worst_col = missing.idxmax() if len(missing) > 0 else "none"
    chk("C9_feature_missing_rate",
        worst <= 0.60,
        f"Max missing rate={worst:.1%} in '{worst_col}' (need <=60%)")

    return checks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=== TRUST-CT Phase 2: Cimas HTN Landmark Cohort Builder ===\n")

    # ---- Load raw data ----
    print(f"Loading raw claims: {RAW_CSV.name}")
    raw = _load_raw(RAW_CSV)
    print(f"  Raw: {len(raw):,} rows x {raw.shape[1]} cols")

    print(f"Loading plan info: {ADDL_CSV.name}")
    plan_info = _load_plan(ADDL_CSV)
    print(f"  Plans: {len(plan_info)} rows")

    # ---- Clean ----
    print("\nCleaning claims...")
    claims = build_clean_claims(raw)
    print(f"  After 2022 filter: {len(claims):,} claims, {claims['pid'].nunique():,} PIDs")

    # Save clean claims
    claims.to_parquet(OUT_DIR / "cimas_htn_claims_clean.parquet", index=False)
    print(f"  Saved: cimas_htn_claims_clean.parquet")

    # ---- Inclusion filters ----
    print("\nApplying inclusion filters...")
    all_2022, obs, flow = apply_inclusion(claims)
    print(f"  Obs window PIDs: {flow['obs_window_pids']:,}")

    # ---- Outcome ----
    print("\nComputing Y_adh from out window (Jul-Dec 2022)...")
    outcome = compute_outcome(all_2022)
    print(f"  PIDs with out-window claims: {len(outcome):,}")
    print(f"  Y_adh=1: {outcome['Y_adh'].sum():,} ({outcome['Y_adh'].mean():.1%})")
    flow["out_window_pids"]      = outcome["pid"].nunique()
    flow["y_adh_positive"]       = int(outcome["Y_adh"].sum())
    flow["y_adh_prevalence_out"] = float(outcome["Y_adh"].mean())

    # ---- Features ----
    print("\nComputing landmark features from obs window...")
    features = compute_features(obs, plan_info)
    print(f"  Feature rows: {len(features):,}, cols: {features.shape[1]}")

    # ---- Merge features + outcome ----
    # All PIDs with obs window features; fill Y_adh=0 for those absent from out window
    landmark = features.merge(outcome, on="pid", how="left")
    landmark["out_refill_months"] = landmark["out_refill_months"].fillna(0).astype(int)
    landmark["Y_adh"]             = landmark["Y_adh"].fillna(0).astype(int)
    flow["landmark_n"] = len(landmark)
    flow["y_adh_prevalence_landmark"] = float(landmark["Y_adh"].mean())
    print(f"  Landmark cohort: {len(landmark):,} PIDs")
    print(f"  Y_adh prevalence (landmark): {landmark['Y_adh'].mean():.1%}")

    # ---- Client assignment ----
    print("\nAssigning clients by first dispensing provider...")
    client_df, manifest = assign_clients(obs)
    landmark = landmark.merge(client_df, on="pid", how="left")
    landmark["client"]      = landmark["client"].fillna("other")
    landmark["client_sens"] = landmark["client_sens"].fillna("other")

    n_primary_clients = manifest[manifest["client_primary"]]["assigned_provider"].nunique()
    flow["n_primary_clients"]     = int(n_primary_clients)
    flow["n_sensitivity_clients"] = int(manifest[manifest["client_sensitivity"]]["assigned_provider"].nunique())
    print(f"  Primary clients (>=100 PIDs): {n_primary_clients}")
    print(f"  Sensitivity clients (>=50 PIDs): {flow['n_sensitivity_clients']}")

    # Save landmark dataset
    landmark.to_parquet(OUT_DIR / "cimas_htn_landmark_6m.parquet", index=False)
    print(f"\n  Saved: cimas_htn_landmark_6m.parquet")

    # Save provider manifest
    manifest_path = OUT_DIR / "cimas_provider_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"  Saved: cimas_provider_manifest.csv")

    # ---- Automated checks ----
    print("\n--- Automated Quality Checks ---")
    checks = run_checks(landmark, manifest, flow)
    n_pass = sum(c["passed"] for c in checks)
    n_fail = len(checks) - n_pass
    print(f"\n  Result: {n_pass}/{len(checks)} checks passed", end="")
    if n_fail > 0:
        print(f", {n_fail} FAILED")
    else:
        print(" -- ALL PASS")

    # ---- Save cohort flow ----
    cohort_flow = {
        "flow": flow,
        "checks": checks,
        "feature_columns": [
            c for c in landmark.columns
            if c not in {"pid", "Y_adh", "out_refill_months",
                          "client", "client_sens", "assigned_provider",
                          "ANNUALCONTRIBUTION", "option_name"}
        ],
        "outcome": {
            "name": "Y_adh",
            "definition": ">=5 refill months in Jul-Dec 2022",
            "NOT_PDC": True,
            "NOT_ingestion_confirmed": True,
        },
        "seeds": SEEDS,
    }
    flow_path = OUT_DIR / "cimas_cohort_flow.json"
    with open(flow_path, "w", encoding="utf-8") as f:
        json.dump(cohort_flow, f, indent=2, default=str)
    print(f"  Saved: cimas_cohort_flow.json")

    # ---- Summary table ----
    print("\n--- Feature Summary ---")
    feat_cols = cohort_flow["feature_columns"]
    if feat_cols:
        summary = landmark[feat_cols].agg(["mean", "std", lambda x: x.isna().mean()])
        summary.index = ["mean", "std", "missing_rate"]
        # Trim for display
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.max_columns", 30):
            print(summary.T.to_string())

    # ---- Client breakdown ----
    print("\n--- Client Breakdown (Primary, >=100 PIDs) ---")
    prim = manifest[manifest["client_primary"]].copy()
    prim_merged = prim.merge(
        landmark.groupby("client")["Y_adh"].agg(["sum", "count", "mean"]).reset_index(),
        left_on="assigned_provider", right_on="client", how="left"
    )
    if len(prim_merged) > 0:
        prim_merged = prim_merged.rename(columns={"sum": "n_adherent", "count": "n_pid", "mean": "prev"})
        cols_to_show = ["assigned_provider", "n_pids", "n_adherent", "prev"]
        cols_to_show = [c for c in cols_to_show if c in prim_merged.columns]
        print(prim_merged[cols_to_show].to_string(index=False))

    print("\n=== Build complete ===")
    return landmark, manifest, cohort_flow


if __name__ == "__main__":
    landmark, manifest, flow_data = main()
