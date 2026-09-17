"""
Builds patient_features.csv (landmark, index-implant-time covariates) and
patient_labels.csv (interval-censored time-to-event target) for Head A from
labels.csv, per Master Prompt Section 5.2/5.4.

Landmark = the index implant event itself (age/sex/approach/valve
model/size — everything in Section 5.1's extractor list #1-5 is, by
construction, known at implant, i.e. always <=landmark; there is no
separate "30-day post-implant echo" landmark available in this notes-only
extract, since baseline echo is recoverable for only a minority of
patients — see data_plan.md). No post-landmark note is ever read for a
FEATURE here — only for the label (redo_note_year / last_note_year), which
is the leakage boundary Hard Rule #2 requires.

Cohort restriction (documented, not silently applied): patients with
index_implant_source == 'undated' (18/117) have no time origin at all and
are EXCLUDED from Head A — a durability model cannot place them on a time
axis. This is reported, not hidden (see reports/head_a_cohort_flow.md).

Interval censoring (Master Prompt Section 0/8): year-only dates mean every
event time is only known to fall in [T-1, T+1] where T = event_note_year -
index_implant_year. Right-censored patients contribute their observed
follow-up time (last_note_year - index_implant_year) as a lower bound only.
"""
from __future__ import annotations

import pandas as pd
import yaml

with open("config/priors.yaml", encoding="utf-8") as f:
    PRIORS = yaml.safe_load(f)
VALVE_MODEL_TO_FAMILY = PRIORS["valve_model_to_family"]

# PPM proxy (Section 5.1, extractor #8): true iEOA requires BSA, which is
# not recoverable from these notes; Master Prompt explicitly specifies the
# documented proxy "labelled size <=21mm", flagged as a proxy not a
# measured value.
PPM_PROXY_SIZE_MM = 21


def build_patient_features(labels_path: str = "labels.csv") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(labels_path)
    flow = {"total_patients": len(df)}

    has_index_time = df["index_implant_source"] != "undated"
    flow["excluded_no_index_time"] = int((~has_index_time).sum())
    cohort = df[has_index_time].copy()
    flow["head_a_cohort"] = len(cohort)

    cohort["valve_family"] = cohort["index_valve_model"].map(VALVE_MODEL_TO_FAMILY)
    # Unmapped model (incl. Trifecta, Biocor, Epic, Konect, homograft, or
    # model never recovered at all) falls back to the approach-level
    # population prior at the modeling stage — never silently dropped, and
    # never assigned a fabricated family.
    cohort["valve_family_known"] = cohort["valve_family"].notna()
    cohort["ppm_proxy_flag"] = cohort["index_valve_size_mm"] <= PPM_PROXY_SIZE_MM
    cohort["size_known"] = cohort["index_valve_size_mm"].notna()

    # --- time-to-event construction -----------------------------------
    is_event = cohort["bvf_stage"] == 2
    t_event = cohort["redo_note_year"] - cohort["index_implant_year"]
    t_censor = cohort["last_note_year"] - cohort["index_implant_year"]

    cohort["event"] = is_event.astype(int)
    # Interval-censored bounds in years since index implant. Event: [T-1,
    # T+1], clipped at the event's own lower bound of 0 (can't have had the
    # reintervention before the valve existed) and additionally floored so
    # T_lower is never negative. Right-censored: lower bound = observed
    # follow-up, upper bound = +inf (represented as None / np.inf
    # downstream).
    cohort["t_lower"] = None
    cohort["t_upper"] = None
    cohort.loc[is_event, "t_lower"] = (t_event[is_event] - 1).clip(lower=0)
    cohort.loc[is_event, "t_upper"] = t_event[is_event] + 1
    cohort.loc[~is_event, "t_lower"] = t_censor[~is_event].clip(lower=0)
    cohort.loc[~is_event, "t_upper"] = float("inf")

    feature_cols = [
        "profile_key", "index_implant_year", "index_implant_source", "index_approach",
        "index_valve_model", "valve_family", "valve_family_known",
        "index_valve_size_mm", "size_known", "ppm_proxy_flag",
    ]
    label_cols = ["profile_key", "event", "t_lower", "t_upper", "confidence_tier", "hvd_stage", "bvf_stage"]

    features_df = cohort[feature_cols].reset_index(drop=True)
    labels_out_df = cohort[label_cols].reset_index(drop=True)

    flow["events"] = int(cohort["event"].sum())
    flow["censored"] = int((cohort["event"] == 0).sum())
    flow["events_by_approach"] = cohort.loc[cohort["event"] == 1, "index_approach"].value_counts().to_dict()
    flow["missing_valve_family"] = int((~cohort["valve_family_known"]).sum())
    flow["missing_size"] = int((~cohort["size_known"]).sum())
    flow["age_available"] = False  # [AGE] is masked throughout the corpus — see note below
    flow["age_note"] = (
        "Age at implant is masked ([AGE]) in every note in this corpus with no numeric value "
        "surviving de-identification anywhere it was checked; age cannot be used as a covariate "
        "or as a comparator model (Master Prompt's own 'age-only Weibull' comparator, Section 5.4, "
        "cannot be built for this reason — documented as a limitation, not silently substituted)."
    )
    return features_df, labels_out_df, flow


if __name__ == "__main__":
    features_df, labels_df, flow = build_patient_features()
    features_df.to_csv("data_processed_patient_features.csv", index=False)
    labels_df.to_csv("data_processed_patient_labels.csv", index=False)
    print(yaml.safe_dump(flow, sort_keys=False, allow_unicode=True))
