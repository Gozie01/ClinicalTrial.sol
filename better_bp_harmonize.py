"""
BETTER-BP P1 Harmonization Script
Produces canonical datasets and audit report from the three BETTER-BP REDCap exports.

Decisions fixed by roadmap:
- Site A (n=312 randomized) and Site B (n=90 randomized) are the two FL clients.
- Site C has one non-randomized screen failure and is excluded from all modelling.
- Daily adherence cannot be reconstructed; primary outcome is 12-month visit nonattendance.
- The 402-vs-400 randomized-count discrepancy is flagged but not resolved here.

Outputs (processed/better_bp/):
  participants_baseline.parquet
  visits.parquet
  trial_outcomes.parquet
  bottle_setup.parquet
  medications.parquet
  laboratory_results.parquet
  hospitalizations.parquet
  field_mapping.csv
  cohort_flow.csv
  data_quality_report.json
  data_manifest.json
"""

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "New" / "BETTER-BP"
OUT_DIR = ROOT / "processed" / "better_bp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Site configuration ──────────────────────────────────────────────────────
SITE_FILES = {
    "A": RAW_DIR / "BETTERBP_De-identification-DATA_LABELS_SiteA.xlsx",
    "B": RAW_DIR / "BETTERBP_De-identification-DATA_LABELS_SiteB.xlsx",
    "C": RAW_DIR / "BETTERBP_De-identification-DATA_LABELS_SiteC.xlsx",
}

# Expected counts from user audit (used in mandatory checks)
EXPECTED = {
    "A": {"baseline": 381, "randomized": 312, "visit_6m": 273, "visit_12m": 248},
    "B": {"baseline": 111, "randomized": 90, "visit_6m": 71, "visit_12m": 65},
    "C": {"baseline": 1, "randomized": 0},
}
EXPECTED_TOTAL_RANDOMIZED = 402  # raw count in files (not the published 400)
EXPECTED_INTERVENTION = 266
EXPECTED_CONTROL = 136

# ── Column name aliases across sites ────────────────────────────────────────
# Some columns have slightly different names between Site A and Site B.
COMPLETION_COL_ALIASES = [
    "Did the patient complete the study?",
    "Did the participant complete the study?",
]
EVENT_NAME_COL = "Event Name"
REPEAT_INSTRUMENT_COL = "Repeat Instrument"
RECORD_ID_COL = "Record ID"

# ── Canonical column definitions ─────────────────────────────────────────────
BASELINE_COLS = {
    "randomized": "Randomized?",
    "treatment_arm": "Randomization",
    "sex": "Sex ",
    "age": "Age",
    "ethnicity": "Ethnicity  ",
    "sbp_avg": "Systolic Blood Pressure Avg",
    "dbp_avg": "Diastolic Blood Pressure  Avg",
    "mases_total": "Calculated MASES Total score",
    "bottle_received": "Did the patient receive an AdhereTech bottle? ",
    "bottle_able_open": "Was the participant able to open the bottle? ",
    "hospital_admission_baseline": "Have you been admitted to the hospital since your enrollment visit? ",
}

LAB_CHOICES = [
    "HEMOGLOBIN", "PLATELET COUNT", "WHITE BLOOD CELL COUNT",
    "SODIUM", "POTASSIUM", "BUN", "CREATININE", "GLUCOSE", "TROPONIN",
]

# ── Utility functions ────────────────────────────────────────────────────────

def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise_event(event: str | None) -> str:
    if event is None:
        return ""
    return str(event).strip()


def load_site(path: Path, site_id: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df["_site"] = site_id
    # Normalise key structural columns
    df["Event Name"] = df["Event Name"].apply(normalise_event)
    df["Repeat Instrument"] = df.get("Repeat Instrument", pd.Series(dtype=str)).fillna("")
    df["Record ID"] = df["Record ID"].str.strip()
    # Add a prefixed participant ID to guarantee cross-site uniqueness
    df["participant_id"] = site_id + "_" + df["Record ID"]
    return df


def first_non_null(df: pd.DataFrame, col: str):
    """Return the first non-null value in column col across the rows of df."""
    if col not in df.columns:
        return None
    vals = df[col].dropna()
    return vals.iloc[0] if len(vals) > 0 else None


def numeric(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# ── Per-site processing ──────────────────────────────────────────────────────

def process_site(df: pd.DataFrame, site_id: str) -> dict:
    """
    Returns a dict of DataFrames for this site:
      participants_baseline, visits, trial_outcomes,
      bottle_setup, medications, laboratory_results, hospitalizations
    """
    # Resolve study-completion column (name differs between sites)
    completion_col = None
    for alias in COMPLETION_COL_ALIASES:
        if alias in df.columns:
            completion_col = alias
            break

    # Main-form rows only (Repeat Instrument is blank/NaN)
    main = df[df["Repeat Instrument"] == ""].copy()
    meds_rows = df[df["Repeat Instrument"] == "Medication List"].copy()
    labs_rows = df[df["Repeat Instrument"] == "Lab Values"].copy()
    hosp_rows = df[df["Repeat Instrument"] == "Hospital Admissions"].copy()

    # ── Participants: one row per participant from Baseline main form ──────
    baseline_main = main[main["Event Name"] == "Baseline"].copy()

    records = []
    for pid, grp in baseline_main.groupby("participant_id"):
        row = grp.iloc[0]

        def _get(col_name):
            return first_non_null(grp, col_name)

        rec = {
            "participant_id": pid,
            "site": site_id,
            "record_id_raw": row["Record ID"],
            "randomized": _get(BASELINE_COLS["randomized"]),
            "treatment_arm": _get(BASELINE_COLS["treatment_arm"]),
            "sex": _get(BASELINE_COLS["sex"]),
            "age": numeric(_get(BASELINE_COLS["age"])),
            "ethnicity": _get(BASELINE_COLS["ethnicity"]),
            "sbp_baseline": numeric(_get(BASELINE_COLS["sbp_avg"])),
            "dbp_baseline": numeric(_get(BASELINE_COLS["dbp_avg"])),
            "mases_baseline": numeric(_get(BASELINE_COLS["mases_total"])),
            "bottle_received": _get(BASELINE_COLS["bottle_received"]),
            "bottle_able_open": _get(BASELINE_COLS["bottle_able_open"]),
        }
        # Race columns (choice=...)
        race_choices = [c for c in grp.columns if c.startswith("Race  (choice=")]
        for rc in race_choices:
            race_label = rc.replace("Race  (choice=", "").rstrip(")")
            rec[f"race_{race_label.lower().replace(' ', '_')}"] = _get(rc)

        records.append(rec)

    participants_df = pd.DataFrame(records)

    # ── Visits: one row per participant × visit (6m, 12m) ─────────────────
    visit_events = {
        "6m": "6-Month Follow-up",
        "12m": "12-Month Follow-up",
    }
    visit_records = []
    for visit_key, ev_name in visit_events.items():
        visit_main = main[main["Event Name"] == ev_name].copy()
        for pid, grp in visit_main.groupby("participant_id"):
            rec = {
                "participant_id": pid,
                "site": site_id,
                "visit": visit_key,
                "attended": 1,
                "sbp_visit": numeric(first_non_null(grp, BASELINE_COLS["sbp_avg"])),
                "dbp_visit": numeric(first_non_null(grp, BASELINE_COLS["dbp_avg"])),
                "mases_visit": numeric(first_non_null(grp, BASELINE_COLS["mases_total"])),
            }
            visit_records.append(rec)
    visits_df = pd.DataFrame(visit_records)

    # ── Trial outcomes ─────────────────────────────────────────────────────
    eos_main = main[main["Event Name"].str.startswith("End of Study")].copy()
    eos_map = {}
    for pid, grp in eos_main.groupby("participant_id"):
        comp = first_non_null(grp, completion_col) if completion_col else None
        eos_map[pid] = comp

    # Which randomized participants attended each visit
    visited_6m = set(visits_df[visits_df["visit"] == "6m"]["participant_id"].tolist())
    visited_12m = set(visits_df[visits_df["visit"] == "12m"]["participant_id"].tolist())

    # 52 participants with explicit loss-to-follow-up (from user audit, cannot recover exact from data
    # without discontinuation reason column; mark as missing_eos instead)
    outcome_records = []
    rand_mask = participants_df["randomized"] == "Yes"
    for _, prow in participants_df[rand_mask].iterrows():
        pid = prow["participant_id"]
        comp = eos_map.get(pid)
        has_eos = pid in eos_map
        outcome_records.append({
            "participant_id": pid,
            "site": site_id,
            "treatment_arm": prow["treatment_arm"],
            "visit_6m_attended": int(pid in visited_6m),
            "visit_12m_attended": int(pid in visited_12m),
            "study_completed": int(comp == "Yes") if comp is not None else None,
            "explicit_noncompletion": int(comp == "No") if comp is not None else None,
            "missing_eos_record": int(not has_eos),
            # Primary outcome: 12-month visit nonattendance
            "nonattendance_12m": int(pid not in visited_12m),
            # 6-month nonattendance (for treatment-response analysis)
            "nonattendance_6m": int(pid not in visited_6m),
        })
    outcomes_df = pd.DataFrame(outcome_records)

    # ── BP and MASES changes (joined from visits and baseline) ─────────────
    if not visits_df.empty and not participants_df.empty:
        base_bp = participants_df[["participant_id", "sbp_baseline", "dbp_baseline", "mases_baseline"]].copy()
        for visit_key in ["6m", "12m"]:
            vdf = visits_df[visits_df["visit"] == visit_key][["participant_id", "sbp_visit", "dbp_visit", "mases_visit"]].copy()
            merged = base_bp.merge(vdf, on="participant_id", how="inner")
            if not merged.empty:
                sbp_col_name = f"sbp_change_{visit_key}"
                dbp_col_name = f"dbp_change_{visit_key}"
                mases_col_name = f"mases_change_{visit_key}"
                merged[sbp_col_name] = merged["sbp_visit"] - merged["sbp_baseline"]
                merged[dbp_col_name] = merged["dbp_visit"] - merged["dbp_baseline"]
                merged[mases_col_name] = merged["mases_visit"] - merged["mases_baseline"]
                change_cols = ["participant_id", sbp_col_name, dbp_col_name, mases_col_name]
                outcomes_df = outcomes_df.merge(merged[change_cols], on="participant_id", how="left")

    # ── Bottle setup ───────────────────────────────────────────────────────
    bottle_main = main[main["Event Name"] == "Adhere Tech Bottle"].copy()
    bottle_records = []
    bottle_col_received = BASELINE_COLS["bottle_received"]
    bottle_col_open = BASELINE_COLS["bottle_able_open"]
    bottle_col_returned = "Was the AdhereTech bottle returned?"
    for pid, grp in bottle_main.groupby("participant_id"):
        bottle_records.append({
            "participant_id": pid,
            "site": site_id,
            "bottle_received": first_non_null(grp, bottle_col_received),
            "able_to_open": first_non_null(grp, bottle_col_open),
            "bottle_returned": first_non_null(grp, bottle_col_returned),
        })
    bottle_df = pd.DataFrame(bottle_records)

    # ── Medications (from Baseline main form — columns "What is the baseline medication name?")
    # Note: Medication List repeated-instrument rows carry per-visit medication changes,
    # not the baseline medication names. Baseline medication names are on the main Baseline form.
    med_name_cols = [c for c in baseline_main.columns if "baseline medication name" in c.lower()]
    med_class_col = next((c for c in baseline_main.columns if "baseline medication classification" in c.lower()), None)
    med_records = []
    for _, brow in baseline_main.iterrows():
        pid = brow.get("participant_id")
        if not pid:
            continue
        for col in med_name_cols:
            val = brow.get(col)
            if val and str(val).strip() not in ("", "nan"):
                med_records.append({
                    "participant_id": pid,
                    "site": site_id,
                    "event": "Baseline",
                    "medication_name": str(val).strip(),
                    "medication_class": str(brow.get(med_class_col, "")).strip() if med_class_col else None,
                })
    medications_df = pd.DataFrame(med_records).drop_duplicates() if med_records else pd.DataFrame(
        columns=["participant_id", "site", "event", "medication_name", "medication_class"]
    )

    # ── Lab values ─────────────────────────────────────────────────────────
    lab_records = []
    lab_choice_cols = {
        lab: next((c for c in labs_rows.columns if f"choice={lab}" in c), None)
        for lab in LAB_CHOICES
    }
    for _, lrow in labs_rows.iterrows():
        pid = lrow.get("participant_id")
        ev = str(lrow.get("Event Name", "")).strip()
        if pid:
            rec = {"participant_id": pid, "site": site_id, "event": ev}
            for lab, col in lab_choice_cols.items():
                rec[lab.lower().replace(" ", "_")] = numeric(lrow.get(col)) if col else None
            lab_records.append(rec)
    labs_df = pd.DataFrame(lab_records) if lab_records else pd.DataFrame(
        columns=["participant_id", "site", "event"] + [l.lower().replace(" ", "_") for l in LAB_CHOICES]
    )

    # ── Hospitalizations ───────────────────────────────────────────────────
    hosp_col_admitted = "Have you been admitted to the hospital since your enrollment visit? "
    hosp_records = []
    for _, hrow in hosp_rows.iterrows():
        pid = hrow.get("participant_id")
        ev = str(hrow.get("Event Name", "")).strip()
        if pid:
            hosp_records.append({
                "participant_id": pid,
                "site": site_id,
                "event": ev,
                "admitted_since_enrollment": hrow.get(hosp_col_admitted),
            })
    hosp_df = pd.DataFrame(hosp_records) if hosp_records else pd.DataFrame(
        columns=["participant_id", "site", "event", "admitted_since_enrollment"]
    )

    return {
        "participants_baseline": participants_df,
        "visits": visits_df,
        "trial_outcomes": outcomes_df,
        "bottle_setup": bottle_df,
        "medications": medications_df,
        "laboratory_results": labs_df,
        "hospitalizations": hosp_df,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_p1():
    audit_flags = []
    file_hashes = {}
    per_site_counts = {}

    # Load all three sites
    site_dfs = {}
    for site_id, path in SITE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        file_hashes[f"site_{site_id}"] = file_md5(path)
        site_dfs[site_id] = load_site(path, site_id)

    # Process modelling sites (A and B only)
    all_tables = defaultdict(list)
    for site_id in ("A", "B"):
        tables = process_site(site_dfs[site_id], site_id)
        for table_name, df in tables.items():
            all_tables[table_name].append(df)
        per_site_counts[site_id] = {
            "baseline_records": len(tables["participants_baseline"]),
            "randomized": int((tables["participants_baseline"]["randomized"] == "Yes").sum()),
            "visit_6m": int((tables["trial_outcomes"]["visit_6m_attended"] == 1).sum()) if not tables["trial_outcomes"].empty else 0,
            "visit_12m": int((tables["trial_outcomes"]["visit_12m_attended"] == 1).sum()) if not tables["trial_outcomes"].empty else 0,
        }

    # Audit Site C (excluded from modelling)
    site_c_df = site_dfs["C"]
    c_baseline_main = site_c_df[
        (site_c_df["Event Name"] == "Baseline") & (site_c_df["Repeat Instrument"] == "")
    ]
    c_randomized = int((c_baseline_main.get("Randomized?", pd.Series(dtype=str)) == "Yes").sum())
    per_site_counts["C"] = {
        "baseline_records": len(c_baseline_main),
        "randomized": c_randomized,
    }

    # Concatenate across sites A + B
    combined = {name: pd.concat(dfs, ignore_index=True) for name, dfs in all_tables.items()}

    # ── Mandatory P1 checks ───────────────────────────────────────────────
    checks = {}

    # Check 1: Site A baseline records
    checks["C1_site_a_baseline"] = {
        "description": "Site A: 381 baseline records",
        "expected": EXPECTED["A"]["baseline"],
        "actual": per_site_counts["A"]["baseline_records"],
        "pass": per_site_counts["A"]["baseline_records"] == EXPECTED["A"]["baseline"],
    }

    # Check 2: Site A randomized count
    checks["C2_site_a_randomized"] = {
        "description": "Site A: 312 randomized participants",
        "expected": EXPECTED["A"]["randomized"],
        "actual": per_site_counts["A"]["randomized"],
        "pass": per_site_counts["A"]["randomized"] == EXPECTED["A"]["randomized"],
    }

    # Check 3: Site B baseline records
    checks["C3_site_b_baseline"] = {
        "description": "Site B: 111 baseline records",
        "expected": EXPECTED["B"]["baseline"],
        "actual": per_site_counts["B"]["baseline_records"],
        "pass": per_site_counts["B"]["baseline_records"] == EXPECTED["B"]["baseline"],
    }

    # Check 4: Site B randomized count
    checks["C4_site_b_randomized"] = {
        "description": "Site B: 90 randomized participants",
        "expected": EXPECTED["B"]["randomized"],
        "actual": per_site_counts["B"]["randomized"],
        "pass": per_site_counts["B"]["randomized"] == EXPECTED["B"]["randomized"],
    }

    # Check 5: Site C has 1 baseline and 0 randomized
    checks["C5_site_c_excluded"] = {
        "description": "Site C: 1 baseline (screen failure), 0 randomized",
        "expected_baseline": EXPECTED["C"]["baseline"],
        "expected_randomized": EXPECTED["C"]["randomized"],
        "actual_baseline": per_site_counts["C"]["baseline_records"],
        "actual_randomized": per_site_counts["C"]["randomized"],
        "pass": (
            per_site_counts["C"]["baseline_records"] == EXPECTED["C"]["baseline"]
            and per_site_counts["C"]["randomized"] == EXPECTED["C"]["randomized"]
        ),
    }

    # Check 6: Total randomized = 402 (raw file count; not the published 400)
    total_rand = per_site_counts["A"]["randomized"] + per_site_counts["B"]["randomized"]
    checks["C6_total_randomized"] = {
        "description": "Total randomized across A+B = 402 (raw; published=400, discrepancy flagged)",
        "expected": EXPECTED_TOTAL_RANDOMIZED,
        "actual": total_rand,
        "pass": total_rand == EXPECTED_TOTAL_RANDOMIZED,
        "audit_flag": "402 vs published 400: two-record discrepancy unresolved; do not remove records",
    }

    # Check 7: Arm counts
    outcomes = combined["trial_outcomes"]
    arm_counts = outcomes["treatment_arm"].value_counts().to_dict() if not outcomes.empty else {}
    checks["C7_arm_counts"] = {
        "description": "Intervention=266, Control=136",
        "expected_intervention": EXPECTED_INTERVENTION,
        "expected_control": EXPECTED_CONTROL,
        "actual_intervention": arm_counts.get("Intervention", 0),
        "actual_control": arm_counts.get("Control", 0),
        "pass": (
            arm_counts.get("Intervention", 0) == EXPECTED_INTERVENTION
            and arm_counts.get("Control", 0) == EXPECTED_CONTROL
        ),
    }

    # Check 8: No participant ID appears in both sites
    pids_a = set(all_tables["participants_baseline"][0]["participant_id"].tolist())
    pids_b = set(all_tables["participants_baseline"][1]["participant_id"].tolist())
    cross_site_overlap = pids_a & pids_b
    checks["C8_no_cross_site_pid_overlap"] = {
        "description": "No participant ID occurs in both sites",
        "overlap_count": len(cross_site_overlap),
        "pass": len(cross_site_overlap) == 0,
    }

    # Check 9: Repeated instrument rows not counted as participants
    # Verify that participants_baseline has one row per participant
    baseline_df = combined["participants_baseline"]
    dup_pids = baseline_df[baseline_df.duplicated("participant_id", keep=False)]["participant_id"].unique()
    checks["C9_one_row_per_participant"] = {
        "description": "participants_baseline has exactly one row per participant_id",
        "duplicate_pids": list(dup_pids[:10]),
        "pass": len(dup_pids) == 0,
    }

    # Check 10: No outcome column contains 6m/12m visit or completion data at baseline
    baseline_feature_cols = [
        c for c in baseline_df.columns
        if any(x in str(c).lower() for x in ["6m", "12m", "nonattendance", "completed"])
    ]
    checks["C10_no_leakage_in_baseline"] = {
        "description": "participants_baseline contains no future-visit or completion columns",
        "suspicious_columns": baseline_feature_cols,
        "pass": len(baseline_feature_cols) == 0,
    }

    all_passed = all(v.get("pass", False) for v in checks.values())
    if not all_passed:
        failed = [k for k, v in checks.items() if not v.get("pass", False)]
        audit_flags.append(f"FAILED checks: {failed}")

    # ── Site A 6-month and 12-month attendance cross-checks ──────────────
    checks["C1b_site_a_visit_6m"] = {
        "description": "Site A: 273 attended 6-month visit",
        "expected": EXPECTED["A"]["visit_6m"],
        "actual": per_site_counts["A"]["visit_6m"],
        "pass": per_site_counts["A"]["visit_6m"] == EXPECTED["A"]["visit_6m"],
    }
    checks["C1c_site_a_visit_12m"] = {
        "description": "Site A: 248 attended 12-month visit",
        "expected": EXPECTED["A"]["visit_12m"],
        "actual": per_site_counts["A"]["visit_12m"],
        "pass": per_site_counts["A"]["visit_12m"] == EXPECTED["A"]["visit_12m"],
    }
    checks["C2b_site_b_visit_6m"] = {
        "description": "Site B: 71 attended 6-month visit",
        "expected": EXPECTED["B"]["visit_6m"],
        "actual": per_site_counts["B"]["visit_6m"],
        "pass": per_site_counts["B"]["visit_6m"] == EXPECTED["B"]["visit_6m"],
    }
    checks["C2c_site_b_visit_12m"] = {
        "description": "Site B: 65 attended 12-month visit",
        "expected": EXPECTED["B"]["visit_12m"],
        "actual": per_site_counts["B"]["visit_12m"],
        "pass": per_site_counts["B"]["visit_12m"] == EXPECTED["B"]["visit_12m"],
    }

    # ── Cohort flow ───────────────────────────────────────────────────────
    cohort_flow_rows = []
    for site_id in ("A", "B", "C"):
        sc = per_site_counts[site_id]
        cohort_flow_rows.append({
            "site": site_id,
            "baseline_records": sc.get("baseline_records", None),
            "randomized": sc.get("randomized", None),
            "visit_6m_attended": sc.get("visit_6m", None),
            "visit_12m_attended": sc.get("visit_12m", None),
            "used_as_fl_client": site_id in ("A", "B"),
        })
    cohort_flow_rows.append({
        "site": "TOTAL (A+B)",
        "baseline_records": per_site_counts["A"]["baseline_records"] + per_site_counts["B"]["baseline_records"],
        "randomized": total_rand,
        "visit_6m_attended": per_site_counts["A"]["visit_6m"] + per_site_counts["B"]["visit_6m"],
        "visit_12m_attended": per_site_counts["A"]["visit_12m"] + per_site_counts["B"]["visit_12m"],
        "used_as_fl_client": True,
    })
    cohort_flow_df = pd.DataFrame(cohort_flow_rows)

    # Add primary outcome summary
    if not outcomes.empty:
        nonatt_12m_total = int(outcomes["nonattendance_12m"].sum())
        nonatt_12m_pct = round(100.0 * nonatt_12m_total / len(outcomes), 1)
        nonatt_6m_total = int(outcomes["nonattendance_6m"].sum())
        nonatt_6m_pct = round(100.0 * nonatt_6m_total / len(outcomes), 1)
        noncomplete_total = int((outcomes["explicit_noncompletion"] == 1).sum())
        missing_eos_total = int((outcomes["missing_eos_record"] == 1).sum())
    else:
        nonatt_12m_total = nonatt_12m_pct = nonatt_6m_total = nonatt_6m_pct = None
        noncomplete_total = missing_eos_total = None

    # ── Field mapping ─────────────────────────────────────────────────────
    field_mapping_rows = [
        {"canonical_name": "participant_id", "source_col": "Record ID (prefixed by site)", "sites": "A,B,C", "transformation": "site + '_' + Record ID", "included": True},
        {"canonical_name": "site", "source_col": "_site (injected)", "sites": "A,B,C", "transformation": "literal site letter", "included": True},
        {"canonical_name": "randomized", "source_col": "Randomized?", "sites": "A,B", "transformation": "raw string Yes/No", "included": True},
        {"canonical_name": "treatment_arm", "source_col": "Randomization", "sites": "A,B", "transformation": "raw string Intervention/Control", "included": True},
        {"canonical_name": "sex", "source_col": "Sex ", "sites": "A,B", "transformation": "raw string", "included": True},
        {"canonical_name": "age", "source_col": "Age", "sites": "A,B", "transformation": "numeric", "included": True},
        {"canonical_name": "ethnicity", "source_col": "Ethnicity  ", "sites": "A,B", "transformation": "raw string", "included": True},
        {"canonical_name": "sbp_baseline", "source_col": "Systolic Blood Pressure Avg", "sites": "A,B", "transformation": "numeric (Baseline event)", "included": True},
        {"canonical_name": "dbp_baseline", "source_col": "Diastolic Blood Pressure  Avg", "sites": "A,B", "transformation": "numeric (Baseline event)", "included": True},
        {"canonical_name": "mases_baseline", "source_col": "Calculated MASES Total score", "sites": "A,B", "transformation": "numeric (Baseline event)", "included": True},
        {"canonical_name": "visit_6m_attended", "source_col": "presence of 6-Month Follow-up main form row", "sites": "A,B", "transformation": "1 if row exists else 0", "included": True},
        {"canonical_name": "visit_12m_attended", "source_col": "presence of 12-Month Follow-up main form row", "sites": "A,B", "transformation": "1 if row exists else 0", "included": True},
        {"canonical_name": "study_completed", "source_col": "Did the patient/participant complete the study?", "sites": "A,B", "transformation": "1 if Yes, 0 if No, None if missing", "included": True},
        {"canonical_name": "explicit_noncompletion", "source_col": "Did the patient/participant complete the study?", "sites": "A,B", "transformation": "1 if No else 0", "included": True},
        {"canonical_name": "missing_eos_record", "source_col": "End of Study event presence", "sites": "A,B", "transformation": "1 if no EoS row found", "included": True},
        {"canonical_name": "nonattendance_12m", "source_col": "derived", "sites": "A,B", "transformation": "1 - visit_12m_attended (primary outcome)", "included": True},
        {"canonical_name": "nonattendance_6m", "source_col": "derived", "sites": "A,B", "transformation": "1 - visit_6m_attended (treatment-response endpoint)", "included": True},
        {"canonical_name": "sbp_change_6m", "source_col": "Systolic Blood Pressure Avg at 6m minus baseline", "sites": "A,B", "transformation": "numeric difference", "included": True},
        {"canonical_name": "sbp_change_12m", "source_col": "Systolic Blood Pressure Avg at 12m minus baseline", "sites": "A,B", "transformation": "numeric difference", "included": True},
        {"canonical_name": "dbp_change_6m", "source_col": "Diastolic Blood Pressure  Avg at 6m minus baseline", "sites": "A,B", "transformation": "numeric difference", "included": True},
        {"canonical_name": "dbp_change_12m", "source_col": "Diastolic Blood Pressure  Avg at 12m minus baseline", "sites": "A,B", "transformation": "numeric difference", "included": True},
        {"canonical_name": "mases_change_6m", "source_col": "Calculated MASES Total score at 6m minus baseline", "sites": "A,B", "transformation": "numeric difference (self-efficacy, NOT objective adherence)", "included": True},
        {"canonical_name": "mases_change_12m", "source_col": "Calculated MASES Total score at 12m minus baseline", "sites": "A,B", "transformation": "numeric difference", "included": True},
        {"canonical_name": "daily_adherence_pct", "source_col": "NOT AVAILABLE", "sites": "none", "transformation": "absent from public release", "included": False},
        {"canonical_name": "lottery_outcome", "source_col": "NOT AVAILABLE", "sites": "none", "transformation": "absent from public release", "included": False},
        {"canonical_name": "incentive_payment_amount", "source_col": "NOT AVAILABLE", "sites": "none", "transformation": "absent from public release", "included": False},
        {"canonical_name": "adequate_adherence_pct_6m", "source_col": "NOT AVAILABLE", "sites": "none", "transformation": "absent from public release — cannot reproduce 71% vs 34%", "included": False},
    ]
    field_mapping_df = pd.DataFrame(field_mapping_rows)

    # ── Data quality report ───────────────────────────────────────────────
    bp_missingness = {}
    for site_id in ("A", "B"):
        site_outcomes = outcomes[outcomes["site"] == site_id] if not outcomes.empty else pd.DataFrame()
        site_baseline = combined["participants_baseline"]
        site_baseline = site_baseline[site_baseline["site"] == site_id]
        bp_missing = int(site_baseline["sbp_baseline"].isna().sum())
        bp_missingness[site_id] = bp_missing

    dq_report = {
        "per_site_counts": per_site_counts,
        "primary_outcome_12m_nonattendance": {
            "n_events": nonatt_12m_total,
            "pct": nonatt_12m_pct,
            "n_total_randomized_ab": total_rand,
        },
        "secondary_outcome_6m_nonattendance": {
            "n_events": nonatt_6m_total,
            "pct": nonatt_6m_pct,
        },
        "explicit_noncompletion_count": noncomplete_total,
        "missing_eos_count": missing_eos_total,
        "bp_baseline_missing_by_site": bp_missingness,
        "mandatory_checks": checks,
        "all_mandatory_checks_passed": all(v.get("pass", False) for v in checks.values()),
        "audit_flags": audit_flags + [
            "402-vs-400 randomized discrepancy: do not remove records without official exclusion rule",
            "MASES is medication adherence SELF-EFFICACY, not objectively measured adherence",
            "AdhereTech Bottle rows are setup/return records, not daily opening events",
            "Formal DP cannot be claimed; clipping+noise mechanism is empirical leakage mitigation only",
            "Site C excluded from modelling (1 non-randomized screen failure)",
        ],
    }

    # ── Data manifest ─────────────────────────────────────────────────────
    manifest = {
        "preprocessing_version": "P1.1",
        "source_files": {k: {"path": str(v), "md5": file_hashes.get(f"site_{k}")} for k, v in SITE_FILES.items()},
        "output_directory": str(OUT_DIR),
        "canonical_tables": {
            "participants_baseline": "one row per participant from Baseline main form",
            "visits": "one row per participant × visit (6m, 12m) for those who attended",
            "trial_outcomes": "one row per randomized participant with all endpoints",
            "bottle_setup": "AdhereTech bottle setup/return records per participant",
            "medications": "Medication List repeated instrument rows (Baseline event)",
            "laboratory_results": "Lab Values repeated instrument rows per visit",
            "hospitalizations": "Hospital Admissions repeated instrument rows per visit",
        },
        "primary_outcome": "nonattendance_12m = 1 if participant has no 12-month visit main form row",
        "treatment_response_endpoint": "nonattendance_6m = 1 if participant has no 6-month visit main form row",
        "fl_clients": {"A": "n=312 randomized", "B": "n=90 randomized"},
        "excluded_sites": {"C": "1 non-randomized screen failure"},
        "unavailable_data": [
            "daily_adherence_pct",
            "adequate_adherence_pct_6m",
            "lottery_outcomes",
            "incentive_payment_amounts",
            "daily_bottle_opening_timestamps",
        ],
        "unresolved_flags": [
            "402 randomized in files vs 400 published; 2-record discrepancy pending official clarification"
        ],
    }

    # ── Save outputs ──────────────────────────────────────────────────────
    for table_name, df in combined.items():
        out_path = OUT_DIR / f"{table_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {table_name}.parquet ({len(df)} rows)")

    cohort_flow_df.to_csv(OUT_DIR / "cohort_flow.csv", index=False)
    field_mapping_df.to_csv(OUT_DIR / "field_mapping.csv", index=False)
    with open(OUT_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(dq_report, f, indent=2, default=str)
    with open(OUT_DIR / "data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=== P1 Audit Summary ===")
    print(f"  Sites A+B randomized total : {total_rand}  (published: 400 — discrepancy flagged)")
    print(f"  Primary outcome events (12m nonattendance): {nonatt_12m_total}/{total_rand} = {nonatt_12m_pct}%")
    print(f"  6-month nonattendance: {nonatt_6m_total}/{total_rand} = {nonatt_6m_pct}%")
    print(f"  Explicit noncompletion (End of Study = No): {noncomplete_total}")
    print(f"  Missing End of Study record: {missing_eos_total}")
    print()
    print("=== Mandatory Check Results ===")
    for key, val in checks.items():
        status = "PASS" if val.get("pass") else "FAIL"
        print(f"  [{status}] {key}: {val.get('description', '')}")
        if not val.get("pass"):
            print(f"         expected={val.get('expected')}, actual={val.get('actual')}")
    print()
    all_ok = all(v.get("pass", False) for v in checks.values())
    print(f"All checks passed: {all_ok}")
    print(f"Outputs written to: {OUT_DIR}")
    return dq_report


if __name__ == "__main__":
    run_p1()
