"""
dashboard/app.py
UC Faculty Wage Transparency -- Interactive Dashboard

Run from project root:
    streamlit run dashboard/app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UC Faculty Wage Transparency",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
NAVY       = "#1f2a44"
CHARCOAL   = "#3a3a3a"
GOLD       = "#b8893a"
BG         = "#fafaf7"
LIGHT_GRAY = "#e8e8e4"
MID_GRAY   = "#c8c8c4"

RANK_COLORS = {
    "Assistant": "#b8893a",
    "Associate": "#4a7fa5",
    "Full":      "#1f2a44",
    "Clinical":  "#8b3a3a",
    "Research":  "#3a7a5a",
}
RANKS = ["Assistant", "Associate", "Full", "Clinical", "Research"]

UCOP = "University of California, Office of the President"


def short_campus(name: str) -> str:
    return (
        name.replace("University of California, ", "UC ")
            .replace("Office of the President", "Office of the Pres.")
    )


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_data():
    ry  = pd.read_parquet(DATA_DIR / "rank_year_summary.parquet")
    rcy = pd.read_parquet(DATA_DIR / "rank_campus_year_summary.parquet")
    bns = pd.read_parquet(DATA_DIR / "salary_distribution_bins.parquet")
    did = pd.read_parquet(DATA_DIR / "did_estimates.parquet")
    return ry, rcy, bns, did


ry, rcy, bins_df, did = load_data()

# Extract DiD baseline post2022 coefficient
_row = did[(did["spec"] == "baseline") & (did["term"] == "post2022")].iloc[0]
COEF      = float(_row["coef"])
SE        = float(_row["se"])
CI_LO     = float(_row["ci_lower"])
CI_HI     = float(_row["ci_upper"])
PVAL      = float(_row["pvalue"])
PCT_BELOW = abs(COEF) * 100   # ~6.34 -> display as 6.3%

# UCOP has 1 salary record in the raw data (2013, Full) and does not meet
# the per-cell threshold in the aggregates. We keep it as a known entity
# so the toggle works; its heatmap cells display as NaN (blank).
_campuses_in_data = sorted(rcy["Campus"].unique())
campuses_teaching = [c for c in _campuses_in_data if c != UCOP]
# Ensure UCOP exists in the campus list even if absent from Parquet
campuses_all      = campuses_teaching + [UCOP]
years_all         = sorted(ry["Year"].unique())

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<style>
  .stApp {{ background-color: {BG}; }}
  .block-container {{ padding-top: 1.8rem; max-width: 1260px; }}
  h1 {{ color: {NAVY}; font-size: 1.55rem; font-weight: 700; line-height: 1.3; }}
  h2, h3, h4 {{ color: {NAVY}; }}
  .callout {{
    background: {NAVY}; color: #ffffff;
    border-radius: 6px; padding: 1rem 1.25rem; margin: 0.5rem 0 1rem 0;
  }}
  .callout p {{ margin: 0.25rem 0; font-size: 0.93rem; line-height: 1.5; }}
  .callout .stat-line {{
    font-size: 0.82rem; opacity: 0.80; margin-top: 0.5rem;
  }}
  .section-rule {{ border: none; border-top: 1px solid {LIGHT_GRAY}; margin: 1.2rem 0; }}
  footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
ABSTRACT = (
    "We examine whether the 2022 increase in the salience of wage information "
    "in the University of California (UC) system corresponds with measurable "
    "changes in within-rank faculty salary dispersion. Using a 13-year repeated "
    "cross-section of approximately 229,000 salary records from 2012 to 2024 -- "
    "covering five faculty ranks across ten UC campuses -- we construct annual "
    "coefficients of variation (CV) for each rank and estimate log-linear OLS "
    "regressions with rank fixed effects, campus fixed effects, and a linear time "
    "trend. We find that mean nominal salaries grew by roughly 52% over the sample "
    "period before declining 6.4% in 2024. The baseline regression yields a "
    "post-2022 coefficient of -0.063 (p < 0.001), indicating wages were "
    "approximately 6.3% below the secular trend after 2022, conditional on rank "
    "and campus. The CV declined to varying degrees for Full, Associate, Research, "
    "and Clinical professors after 2022, with Associate professors showing the "
    "largest drop of approximately 4.6 percentage points. Assistant professors are "
    "an exception: their CV rose by roughly 1.4 percentage points, consistent with "
    "external labor-market pressures. A difference-in-differences interaction model "
    "finds no significant narrowing of the wage gap between Full and Assistant "
    "professors, though Associate and Clinical professors fared relatively better "
    "in the post-period. These patterns are consistent with transparency-driven "
    "internal equity adjustments concentrated within, rather than across, faculty ranks."
)

st.title(
    "Wage Transparency and Within-Rank Salary Compression: "
    "Evidence from UC Faculty Compensation, 2012-2024"
)
st.markdown(
    f"<span style='color:{CHARCOAL}; font-size:0.95rem;'>"
    "Joshua Kenworthy &nbsp;&middot;&nbsp; Agnibha Bhattacharya "
    "&nbsp;&middot;&nbsp; Jackson Wurzer &nbsp;&middot;&nbsp; March 2026"
    "</span>",
    unsafe_allow_html=True,
)

with st.expander("Abstract", expanded=True):
    st.markdown(
        f"<p style='color:{CHARCOAL}; font-size:0.92rem; line-height:1.65;'>"
        f"{ABSTRACT}</p>",
        unsafe_allow_html=True,
    )

pdf_path = Path(__file__).parent.parent / "paper.pdf"
if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download paper (PDF)",
            data=f.read(),
            file_name="wage_transparency_uc_faculty.pdf",
            mime="application/pdf",
        )

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared Plotly layout defaults
# ---------------------------------------------------------------------------
LAYOUT_BASE = dict(
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    font=dict(family="sans-serif", color=CHARCOAL, size=12),
    margin=dict(l=55, r=25, t=55, b=50),
)


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3 = st.tabs([
    "View 1 - Within-Rank CV Trends",
    "View 2 - Salary Distributions",
    "View 3 - Cross-Campus Comparisons",
])


# ===========================================================================
# VIEW 1: Within-Rank CV Trends
# ===========================================================================
with tab1:
    st.markdown("### Within-Rank Coefficient of Variation, 2012-2024")

    ctrl, main = st.columns([1, 3], gap="large")

    with ctrl:
        disagg = st.radio(
            "Disaggregation",
            ["System-wide", "Per campus"],
            index=0,
            key="v1_disagg",
        )
        campus_v1 = None
        if disagg == "Per campus":
            campus_v1 = st.selectbox(
                "Campus",
                campuses_all,
                format_func=short_campus,
                key="v1_campus",
            )
        rank_sel = st.multiselect(
            "Ranks",
            RANKS,
            default=RANKS,
            key="v1_ranks",
        )

    with main:
        fig1 = go.Figure()
        for rank in rank_sel:
            if disagg == "System-wide":
                rd = ry[ry["Rank"] == rank].sort_values("Year")
            else:
                rd = rcy[
                    (rcy["Rank"] == rank) & (rcy["Campus"] == campus_v1)
                ].sort_values("Year")

            if rd.empty:
                continue

            fig1.add_trace(go.Scatter(
                x=rd["Year"],
                y=rd["CV"],
                mode="lines+markers",
                name=rank,
                line=dict(color=RANK_COLORS[rank], width=2.2),
                marker=dict(size=6, symbol="circle"),
                hovertemplate=(
                    f"<b>{rank}</b><br>Year: %{{x}}<br>"
                    "CV: %{y:.2f}%<extra></extra>"
                ),
            ))

        fig1.add_vline(
            x=2022,
            line_dash="dash",
            line_color=CHARCOAL,
            line_width=1.4,
            opacity=0.55,
            annotation_text="2022 threshold",
            annotation_position="top right",
            annotation_font=dict(size=11, color=CHARCOAL),
        )

        fig1.update_layout(
            **LAYOUT_BASE,
            height=420,
            xaxis=dict(
                title="Year",
                tickvals=years_all,
                tickangle=-40,
                gridcolor=LIGHT_GRAY,
                showline=True,
                linecolor=MID_GRAY,
            ),
            yaxis=dict(
                title="Coefficient of Variation (%)",
                gridcolor=LIGHT_GRAY,
                showline=True,
                linecolor=MID_GRAY,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
        )
        st.plotly_chart(fig1, use_container_width=True)

    # DiD callout
    st.markdown(
        f"""<div class="callout">
<p><strong>Difference-in-Differences Estimate &mdash; Baseline Specification</strong></p>
<p>Holding rank, campus, and year constant, log nominal wages are approximately
<strong>{PCT_BELOW:.1f}% lower</strong> in the post-2022 period relative to the
2012-2021 baseline, after rank and campus fixed effects absorb composition.
The negative coefficient is consistent with the within-rank compression visible
in the CV trends above.</p>
<p class="stat-line">
post2022 coefficient: {COEF:.4f} &nbsp;|&nbsp;
HC1 SE: {SE:.4f} &nbsp;|&nbsp;
95% CI: [{CI_LO:.4f}, {CI_HI:.4f}] &nbsp;|&nbsp;
p &lt; 0.001
</p>
</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("Methodology"):
        st.markdown(
            f"""
**What CV measures.** The coefficient of variation is:

$$\\text{{CV}}_{{r,t}} = \\frac{{\\sigma_{{r,t}}}}{{\\mu_{{r,t}}}} \\times 100$$

where $r$ indexes rank and $t$ indexes year. It normalizes within-rank salary
dispersion by the mean, making it comparable across periods with different
average wage levels. A declining CV means salaries are converging relative to
the group mean.

**2022 threshold.** The vertical dashed line marks the onset of the
post-2022 period, which coincides with California's Pay Transparency Act
(effective January 2023) and the historic 2022 UC academic labor strikes.

**Baseline OLS specification:**

$$\\log(\\text{{TotalWages}}_i) = \\beta_0 + \\beta_1 \\cdot \\text{{post2022}}_i
+ \\boldsymbol{{\\beta}}_2 \\cdot \\text{{Rank}}_i
+ \\boldsymbol{{\\beta}}_3 \\cdot \\text{{Campus}}_i
+ \\beta_4 \\cdot \\text{{Year}}_i + \\varepsilon_i$$

$\\text{{post2022}} = 1$ for years 2022-2024, 0 otherwise. HC1
heteroskedasticity-robust standard errors. Reference categories:
Assistant Professor (rank), UC Berkeley (campus).

*Computed on the deduplicated regression panel (N=229,386 for TotalWages).*
"""
        )


# ===========================================================================
# VIEW 2: Salary Distributions
# ===========================================================================
def agg_bins(rank: str, campus: str, year: int) -> pd.DataFrame:
    """Return a 50-bin histogram for the given selection."""
    mask = (bins_df["Rank"] == rank) & (bins_df["Year"] == year)
    if campus != "All campuses":
        mask &= bins_df["Campus"] == campus
    sub = bins_df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["midpoint", "bin_left", "bin_right", "count"])

    sub["midpoint"] = (sub["bin_left"] + sub["bin_right"]) / 2

    if campus == "All campuses":
        g_min = sub["bin_left"].min()
        g_max = sub["bin_right"].max()
        edges = np.linspace(g_min, g_max, 51)
        idx = np.clip(np.digitize(sub["midpoint"].values, edges[:-1]) - 1, 0, 49)
        sub = sub.copy()
        sub["cb"] = idx
        agg = sub.groupby("cb")["count"].sum().reset_index()
        agg["bin_left"]  = edges[agg["cb"]]
        agg["bin_right"] = edges[agg["cb"] + 1]
        agg["midpoint"]  = (agg["bin_left"] + agg["bin_right"]) / 2
        return agg[["midpoint", "bin_left", "bin_right", "count"]]

    return sub[["midpoint", "bin_left", "bin_right", "count"]]


def get_stats(rank: str, campus: str, year: int) -> dict | None:
    if campus == "All campuses":
        row = ry[(ry["Rank"] == rank) & (ry["Year"] == year)]
    else:
        row = rcy[
            (rcy["Rank"] == rank)
            & (rcy["Campus"] == campus)
            & (rcy["Year"] == year)
        ]
    if row.empty:
        return None
    r = row.iloc[0]
    return dict(N=int(r["N"]), Mean=r["Mean"], Median=r["Median"],
                P10=r["P10"], P90=r["P90"], CV=r["CV"])


with tab2:
    st.markdown("### Salary Distribution by Rank, Campus, and Year")

    campus_opts = ["All campuses"] + campuses_all

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(f"**Selection A**")
        r_a  = st.selectbox("Rank",   RANKS,       key="v2ra")
        c_a  = st.selectbox("Campus", campus_opts, key="v2ca",
                            format_func=lambda x: x if x == "All campuses" else short_campus(x))
        y_a  = st.selectbox("Year",   years_all[::-1], index=2, key="v2ya")

    with col_b:
        st.markdown(f"**Selection B &mdash; overlay comparison**")
        overlay = st.checkbox("Enable overlay", value=False, key="v2ov")
        r_b = st.selectbox("Rank ", RANKS, key="v2rb",
                           disabled=not overlay)
        c_b = st.selectbox("Campus ", campus_opts, key="v2cb",
                           format_func=lambda x: x if x == "All campuses" else short_campus(x),
                           disabled=not overlay)
        y_b = st.selectbox("Year ", years_all[::-1], index=0, key="v2yb",
                           disabled=not overlay)

    bins_a = agg_bins(r_a, c_a, y_a)
    lbl_a  = f"{r_a} | {short_campus(c_a) if c_a != 'All campuses' else 'All campuses'} | {y_a}"

    fig2 = go.Figure()
    if not bins_a.empty:
        fig2.add_trace(go.Bar(
            x=bins_a["midpoint"],
            y=bins_a["count"],
            width=bins_a["bin_right"] - bins_a["bin_left"],
            name=lbl_a,
            marker_color=RANK_COLORS.get(r_a, NAVY),
            opacity=0.78,
            hovertemplate="Salary: $%{x:,.0f}<br>Count: %{y:,}<extra></extra>",
        ))

    if overlay:
        bins_b = agg_bins(r_b, c_b, y_b)
        lbl_b  = f"{r_b} | {short_campus(c_b) if c_b != 'All campuses' else 'All campuses'} | {y_b}"
        if not bins_b.empty:
            fig2.add_trace(go.Bar(
                x=bins_b["midpoint"],
                y=bins_b["count"],
                width=bins_b["bin_right"] - bins_b["bin_left"],
                name=lbl_b,
                marker_color=GOLD,
                opacity=0.65,
                hovertemplate="Salary: $%{x:,.0f}<br>Count: %{y:,}<extra></extra>",
            ))

    fig2.update_layout(
        **LAYOUT_BASE,
        barmode="overlay",
        height=380,
        xaxis=dict(title="Total Wages (nominal $)", tickformat="$,.0f",
                   gridcolor=LIGHT_GRAY),
        yaxis=dict(title="Count", gridcolor=LIGHT_GRAY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Summary statistics panel
    st.markdown("**Summary Statistics**")
    n_panels = 2 if overlay else 1
    stat_cols = st.columns(n_panels, gap="large")

    def render_stats(col, stats, label):
        with col:
            st.markdown(f"*{label}*")
            if stats is None:
                st.warning("No data available for this selection.")
                return
            c1, c2, c3 = st.columns(3)
            c1.metric("N", f"{stats['N']:,}")
            c2.metric("Mean", f"${stats['Mean']:,.0f}")
            c3.metric("Median", f"${stats['Median']:,.0f}")
            c4, c5, c6 = st.columns(3)
            c4.metric("P10", f"${stats['P10']:,.0f}")
            c5.metric("P90", f"${stats['P90']:,.0f}")
            c6.metric("CV (%)", f"{stats['CV']:.1f}")

    render_stats(stat_cols[0], get_stats(r_a, c_a, y_a), lbl_a)
    if overlay:
        render_stats(stat_cols[1], get_stats(r_b, c_b, y_b), lbl_b)

    with st.expander("Methodology"):
        st.markdown("""
**Histogram construction.** Salary distributions are pre-computed from the
deduplicated regression panel. Each rank x campus x year cell uses 50
equal-width bins spanning the within-cell salary range. When "All campuses" is
selected, bin counts from individual campuses are re-aggregated onto a common
50-bin grid via bin midpoint assignment.

**Summary statistics** (mean, median, P10, P90) are computed directly on the
panel and are exact, not derived from the binned representation.

*Computed on the deduplicated regression panel (N=229,386 for TotalWages).*
""")


# ===========================================================================
# VIEW 3: Cross-Campus Comparisons
# ===========================================================================
with tab3:
    st.markdown("### Within-Rank Salary Dispersion Across UC Campuses")

    ctrl3, main3 = st.columns([1, 3], gap="large")

    with ctrl3:
        year_v3   = st.slider("Year", 2012, 2024, 2022, step=1, key="v3yr")
        rank_v3   = st.selectbox("Rank (heatmap)", RANKS, index=2, key="v3rk")
        show_ucop = st.checkbox(
            "Include UC Office of the President (non-teaching)",
            value=False,
            key="v3ucop",
        )

    with main3:
        yr_data = rcy[rcy["Year"] == year_v3].copy()

        # --- Panel A: CV Heatmap (campus x rank) ---
        teaching_labels = [short_campus(c) for c in campuses_teaching]

        def campus_cv_row(campus_list):
            rows = []
            for c in campus_list:
                vals = []
                for rk in RANKS:
                    cell = yr_data[(yr_data["Campus"] == c) & (yr_data["Rank"] == rk)]
                    vals.append(float(cell["CV"].values[0]) if not cell.empty else np.nan)
                rows.append(vals)
            return np.array(rows, dtype=float)

        z_teaching = campus_cv_row(campuses_teaching)

        if show_ucop:
            ucop_row = campus_cv_row([UCOP])
            ucop_has_data = not np.all(np.isnan(ucop_row))
            fig3a = make_subplots(
                rows=2, cols=1,
                row_heights=[len(campuses_teaching), 1.6],
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[
                    "10 Teaching Campuses",
                    "UC Office of the President (non-teaching)",
                ],
            )
            fig3a.add_trace(
                go.Heatmap(
                    z=z_teaching,
                    x=RANKS,
                    y=teaching_labels,
                    colorscale=[[0, "#d9e4f0"], [1, NAVY]],
                    showscale=True,
                    colorbar=dict(title="CV (%)", len=0.75, y=0.55),
                    hovertemplate="Campus: %{y}<br>Rank: %{x}<br>CV: %{z:.2f}%<extra></extra>",
                    zmin=np.nanmin(z_teaching),
                    zmax=np.nanmax(z_teaching),
                ),
                row=1, col=1,
            )
            # UCOP row uses gold colorscale; NaN cells render as background
            fig3a.add_trace(
                go.Heatmap(
                    z=ucop_row,
                    x=RANKS,
                    y=["UC Office of the Pres."],
                    colorscale=[[0, "#fdf5e6"], [1, GOLD]],
                    showscale=False,
                    opacity=0.85,
                    hovertemplate=(
                        "Campus: %{y}<br>Rank: %{x}<br>"
                        "CV: %{z:.1f}% (non-teaching)<extra></extra>"
                    ),
                ),
                row=2, col=1,
            )
            fig3a.update_layout(
                **LAYOUT_BASE,
                height=520,
                margin=dict(l=145, r=80, t=65, b=45),
                yaxis=dict(autorange="reversed"),
                yaxis2=dict(autorange="reversed"),
            )
            if not ucop_has_data:
                st.caption(
                    "UC Office of the President appears in the raw dataset with "
                    "fewer than two qualifying salary records in any rank x year cell "
                    "and is displayed with no fill. The paper's 'ten campus' framing "
                    "reflects this absence."
                )
        else:
            fig3a = go.Figure(go.Heatmap(
                z=z_teaching,
                x=RANKS,
                y=teaching_labels,
                colorscale=[[0, "#d9e4f0"], [1, NAVY]],
                showscale=True,
                colorbar=dict(title="CV (%)"),
                hovertemplate="Campus: %{y}<br>Rank: %{x}<br>CV: %{z:.2f}%<extra></extra>",
            ))
            fig3a.update_layout(
                **LAYOUT_BASE,
                height=430,
                margin=dict(l=145, r=80, t=55, b=45),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(title="Rank"),
            )

        st.plotly_chart(fig3a, use_container_width=True)

        # --- Panel B: Campus-level median wages by rank ---
        st.markdown(
            f"**Median Total Wages by Campus and Rank &mdash; {year_v3}**"
        )

        bar_campuses = campuses_teaching + ([UCOP] if show_ucop else [])
        bar_labels   = [short_campus(c) for c in bar_campuses]
        n_t          = len(campuses_teaching)

        fig3b = go.Figure()
        for rank in RANKS:
            meds = []
            for campus in bar_campuses:
                cell = yr_data[(yr_data["Campus"] == campus) & (yr_data["Rank"] == rank)]
                meds.append(float(cell["Median"].values[0]) if not cell.empty else np.nan)

            fig3b.add_trace(go.Bar(
                name=rank,
                x=bar_labels,
                y=meds,
                marker_color=RANK_COLORS[rank],
                opacity=0.88,
                hovertemplate=(
                    f"<b>{rank}</b><br>Campus: %{{x}}<br>"
                    "Median: $%{y:,.0f}<extra></extra>"
                ),
            ))

        if show_ucop and len(bar_campuses) > n_t:
            fig3b.add_vline(
                x=n_t - 0.5,
                line_dash="dot",
                line_color=CHARCOAL,
                line_width=1.2,
                opacity=0.45,
                annotation_text="non-teaching",
                annotation_position="top right",
                annotation_font=dict(size=10, color=CHARCOAL),
            )

        fig3b.update_layout(
            **LAYOUT_BASE,
            barmode="group",
            height=420,
            margin=dict(l=70, r=25, t=45, b=110),
            xaxis=dict(title="Campus", tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(
                title="Median Total Wages ($)",
                tickformat="$,.0f",
                gridcolor=LIGHT_GRAY,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10),
            ),
        )
        st.plotly_chart(fig3b, use_container_width=True)

    with st.expander("Methodology"):
        st.markdown("""
**Heatmap (Panel A).** Each cell shows the within-rank coefficient of variation
($\\text{CV} = 100 \\times \\sigma / \\mu$) for a given campus x rank x year
combination. Darker blue indicates higher within-rank salary dispersion.
Cells with fewer than two observations are blank.

**UC Office of the President** is excluded by default. It is not a teaching
campus and operates under different compensation structures. When enabled,
it appears in a separate gold-toned panel below the ten teaching campuses.

**Median wages (Panel B).** The grouped bar chart shows the within-cell median
nominal total wage for each campus x rank combination in the selected year.
The dotted vertical line (when UCOP is visible) separates the ten teaching
campuses from the non-teaching entity.

*Computed on the deduplicated regression panel (N=229,386 for TotalWages).*
""")


# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown(
    f"""
<div style="color:{CHARCOAL}; font-size:0.80rem; line-height:1.9;">
  <strong>Data source:</strong> UC Annual Wage Database (2012) and California
  State Controller's Government Compensation in California database (2013-2024).
  All data are publicly accessible and contain no individual identifiers.
  &nbsp;|&nbsp;
  <strong>GitHub:</strong>
  <a href="https://github.com/jkenworthy-econ/421-proj"
     style="color:{GOLD};">github.com/jkenworthy-econ/421-proj</a>
  &nbsp;|&nbsp;
  <strong>Contact:</strong> joshkenworthy2002@gmail.com
</div>
""",
    unsafe_allow_html=True,
)
