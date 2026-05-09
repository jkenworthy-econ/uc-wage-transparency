"""
scripts/build_aggregates.py
---------------------------
One-time pre-aggregation pipeline for the UC Faculty Wage Transparency
dashboard. Produces four Parquet files in dashboard/data/ from the
canonical cell-36 panel (deduplicated, TotalWages > 0, N=229,386).

Run from the project root:
    python scripts/build_aggregates.py

Validation: asserts that rank-year CVs match the values committed to
paper.tex (Table 2) within 0.01 percentage points. Raises ValueError
naming any failing cell before writing any Parquet output.
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROF_DIR = ROOT / "professors"
OUT_DIR = ROOT / "dashboard" / "data"

KEEP_COLS = ["Year", "EmployerName", "Position", "RegularPay", "TotalWages"]
RANKS = ["Assistant", "Associate", "Full", "Clinical", "Research"]

# ---------------------------------------------------------------------------
# Paper Table 2 targets (corrected, cell-36 panel, committed 2026-05-09)
# pre = average CV across 2012-2021; post = average CV across 2022-2024
# ---------------------------------------------------------------------------
TABLE2_TARGETS = {
    "Assistant": {"pre": 73.13, "post": 74.55},
    "Associate": {"pre": 76.52, "post": 71.90},
    "Full":      {"pre": 59.22, "post": 56.30},
    "Clinical":  {"pre": 64.81, "post": 64.19},
    "Research":  {"pre": 68.53, "post": 65.64},
}
CV_TOLERANCE = 0.01  # percentage points


# ---------------------------------------------------------------------------
# Step 1: Load panel
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    print("[1] Loading raw CSVs (2012-2024) ...")
    frames = []
    for year in range(2012, 2025):
        path = PROF_DIR / f"uc_professors_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_csv(path)
        for col in KEEP_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[KEEP_COLS].copy()
        df["Year"] = year
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"    {year}: dropped {before - after:,} duplicate rows "
                  f"({before:,} -> {after:,})")
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    print(f"    Combined rows (all years, after dedup): {len(panel):,}")

    pos_lower = panel["Position"].fillna("").str.lower()
    panel["Rank"] = np.select(
        [
            pos_lower.str.contains("assistant") | pos_lower.str.contains("asst"),
            pos_lower.str.contains("associate") | pos_lower.str.contains("assoc"),
            pos_lower.str.contains("clinical")  | pos_lower.str.contains("clin"),
            pos_lower.str.contains("research")  | pos_lower.str.contains("res"),
        ],
        ["Assistant", "Associate", "Clinical", "Research"],
        default="Full",
    )

    panel["TotalWages"] = pd.to_numeric(panel["TotalWages"], errors="coerce")
    panel["RegularPay"] = pd.to_numeric(panel["RegularPay"], errors="coerce")
    panel["post2022"] = (panel["Year"] >= 2022).astype(int)

    panel_tw = panel[panel["TotalWages"] > 0].copy()
    panel_tw["log_total_wages"] = np.log(panel_tw["TotalWages"])

    print(f"    Rows after TotalWages > 0 filter: {len(panel_tw):,}")
    if len(panel_tw) != 229_386:
        raise ValueError(
            f"Expected 229,386 rows in panel_tw, got {len(panel_tw):,}. "
            "Panel construction has changed."
        )

    rank_counts = panel_tw.groupby("Rank").size()
    print("    Row counts by rank:")
    for rank in RANKS:
        print(f"      {rank:12s}: {rank_counts.get(rank, 0):,}")

    return panel_tw


# ---------------------------------------------------------------------------
# Step 2: Validate against paper Table 2 before writing anything
# ---------------------------------------------------------------------------
def validate_cvs(panel_tw: pd.DataFrame) -> dict:
    """
    Compute pre- and post-2022 average CVs by rank and compare to
    TABLE2_TARGETS. Raises ValueError on any cell that exceeds CV_TOLERANCE.
    Returns the full rank x year CV table for use downstream.
    """
    print("\n[2] Validating CVs against paper Table 2 ...")

    # Compute CV for every rank x year cell
    rows = []
    for year in sorted(panel_tw["Year"].unique()):
        for rank in RANKS:
            d = panel_tw.loc[
                (panel_tw["Year"] == year) & (panel_tw["Rank"] == rank),
                "TotalWages",
            ]
            if len(d) < 2:
                continue
            mean_ = d.mean()
            std_ = d.std()
            rows.append({
                "Rank": rank,
                "Year": year,
                "N": len(d),
                "Mean": mean_,
                "Median": d.median(),
                "P10": d.quantile(0.10),
                "P25": d.quantile(0.25),
                "P75": d.quantile(0.75),
                "P90": d.quantile(0.90),
                "Std": std_,
                "CV": std_ / mean_ * 100,
            })

    cv_df = pd.DataFrame(rows)

    # Compute average pre/post CVs and compare to targets
    print(f"\n    {'Rank':12s} {'Pre-2022':>10s} {'Target':>10s} {'Diff':>8s} "
          f"{'Post-2022':>11s} {'Target':>10s} {'Diff':>8s} {'OK?':>5s}")
    print("    " + "-" * 80)

    failures = []
    for rank in RANKS:
        rd = cv_df[cv_df["Rank"] == rank]
        pre_cv  = rd[rd["Year"] < 2022]["CV"].mean()
        post_cv = rd[rd["Year"] >= 2022]["CV"].mean()
        target_pre  = TABLE2_TARGETS[rank]["pre"]
        target_post = TABLE2_TARGETS[rank]["post"]
        diff_pre  = abs(pre_cv  - target_pre)
        diff_post = abs(post_cv - target_post)
        ok = diff_pre <= CV_TOLERANCE and diff_post <= CV_TOLERANCE
        status = "OK" if ok else "FAIL"
        print(f"    {rank:12s} {pre_cv:>10.2f} {target_pre:>10.2f} "
              f"{diff_pre:>+8.4f} {post_cv:>11.2f} {target_post:>10.2f} "
              f"{diff_post:>+8.4f} {status:>5s}")
        if not ok:
            failures.append(
                f"{rank}: pre expected {target_pre}, got {pre_cv:.4f} "
                f"(diff {diff_pre:.4f}); post expected {target_post}, "
                f"got {post_cv:.4f} (diff {diff_post:.4f})"
            )

    if failures:
        raise ValueError(
            "CV validation failed -- do not write Parquet files.\n"
            + "\n".join(failures)
        )

    print("\n    All cells within tolerance. Proceeding to output.\n")
    return cv_df


# ---------------------------------------------------------------------------
# Step 3: Build aggregate tables
# ---------------------------------------------------------------------------
def build_rank_year_summary(cv_df: pd.DataFrame) -> pd.DataFrame:
    print("[3] Building rank_year_summary.parquet ...")
    out = cv_df[["Rank", "Year", "N", "Mean", "Median", "P10", "P25",
                 "P75", "P90", "Std", "CV"]].copy()
    out["post2022"] = (out["Year"] >= 2022).astype(int)
    print(f"    Rows: {len(out):,}  (ranks x years: {len(RANKS)} x 13 = 65)")
    return out


def build_rank_campus_year_summary(panel_tw: pd.DataFrame) -> pd.DataFrame:
    print("[4] Building rank_campus_year_summary.parquet ...")
    rows = []
    for year in sorted(panel_tw["Year"].unique()):
        for rank in RANKS:
            for campus in sorted(panel_tw["EmployerName"].unique()):
                d = panel_tw.loc[
                    (panel_tw["Year"] == year)
                    & (panel_tw["Rank"] == rank)
                    & (panel_tw["EmployerName"] == campus),
                    "TotalWages",
                ]
                if len(d) < 2:
                    continue
                mean_ = d.mean()
                std_ = d.std()
                rows.append({
                    "Rank": rank,
                    "Campus": campus,
                    "Year": year,
                    "N": len(d),
                    "Mean": mean_,
                    "Median": d.median(),
                    "P10": d.quantile(0.10),
                    "P25": d.quantile(0.25),
                    "P75": d.quantile(0.75),
                    "P90": d.quantile(0.90),
                    "Std": std_,
                    "CV": std_ / mean_ * 100,
                })
    out = pd.DataFrame(rows)
    out["post2022"] = (out["Year"] >= 2022).astype(int)
    print(f"    Rows: {len(out):,}")
    return out


def build_salary_bins(panel_tw: pd.DataFrame) -> pd.DataFrame:
    """50 equal-width bins per rank x campus x year cell."""
    print("[5] Building salary_distribution_bins.parquet ...")
    N_BINS = 50
    rows = []
    for year in sorted(panel_tw["Year"].unique()):
        for rank in RANKS:
            for campus in sorted(panel_tw["EmployerName"].unique()):
                d = panel_tw.loc[
                    (panel_tw["Year"] == year)
                    & (panel_tw["Rank"] == rank)
                    & (panel_tw["EmployerName"] == campus),
                    "TotalWages",
                ]
                if len(d) < 5:
                    continue
                counts, edges = np.histogram(d, bins=N_BINS)
                for i in range(N_BINS):
                    rows.append({
                        "Rank": rank,
                        "Campus": campus,
                        "Year": year,
                        "bin_left": edges[i],
                        "bin_right": edges[i + 1],
                        "count": int(counts[i]),
                    })
    out = pd.DataFrame(rows)
    print(f"    Rows: {len(out):,}")
    return out


def build_did_estimates(panel_tw: pd.DataFrame) -> pd.DataFrame:
    """Run both regression specs and store all coefficients with CI."""
    print("[6] Running regressions for did_estimates.parquet ...")

    def extract_estimates(model, spec_label: str) -> pd.DataFrame:
        ci = model.conf_int()
        df = pd.DataFrame({
            "spec":     spec_label,
            "term":     model.params.index,
            "coef":     model.params.values,
            "se":       model.bse.values,
            "tstat":    model.tvalues.values,
            "pvalue":   model.pvalues.values,
            "ci_lower": ci[0].values,
            "ci_upper": ci[1].values,
        })
        df["nobs"] = int(model.nobs)
        return df

    formula_baseline = (
        "log_total_wages ~ post2022 + C(Rank) + C(EmployerName) + Year"
    )
    formula_did = (
        "log_total_wages ~ post2022 * C(Rank) + C(EmployerName) + Year"
    )

    print("    Fitting baseline spec ...")
    model_baseline = smf.ols(formula_baseline, data=panel_tw).fit(cov_type="HC1")
    b_post = model_baseline.params["post2022"]
    b_se   = model_baseline.bse["post2022"]
    print(f"      post2022 coef = {b_post:.4f}  SE = {b_se:.4f}  "
          f"p = {model_baseline.pvalues['post2022']:.4g}")

    print("    Fitting DiD interaction spec ...")
    model_did = smf.ols(formula_did, data=panel_tw).fit(cov_type="HC1")
    print(f"      N = {int(model_did.nobs):,}")

    estimates = pd.concat([
        extract_estimates(model_baseline, "baseline"),
        extract_estimates(model_did, "did"),
    ], ignore_index=True)

    print(f"    Rows: {len(estimates):,}  (coefficients across both specs)")
    return estimates


# ---------------------------------------------------------------------------
# Step 4: Write Parquet files
# ---------------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, name: str):
    path = OUT_DIR / name
    df.to_parquet(path, index=False)
    size_kb = path.stat().st_size / 1024
    print(f"    Written: {path.relative_to(ROOT)}  ({size_kb:.1f} KB, "
          f"{len(df):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("UC Faculty Wage Transparency -- Aggregate Builder")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel_tw = load_panel()
    cv_df    = validate_cvs(panel_tw)   # raises on mismatch; nothing written yet

    rank_year_df         = build_rank_year_summary(cv_df)
    rank_campus_year_df  = build_rank_campus_year_summary(panel_tw)
    bins_df              = build_salary_bins(panel_tw)
    did_df               = build_did_estimates(panel_tw)

    print("\n[7] Writing Parquet files ...")
    write_parquet(rank_year_df,        "rank_year_summary.parquet")
    write_parquet(rank_campus_year_df, "rank_campus_year_summary.parquet")
    write_parquet(bins_df,             "salary_distribution_bins.parquet")
    write_parquet(did_df,              "did_estimates.parquet")

    print("\n" + "=" * 70)
    print("Done. All four Parquet files written to dashboard/data/")
    print("=" * 70)


if __name__ == "__main__":
    main()
