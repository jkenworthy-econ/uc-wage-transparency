UC Professor Salary Compression
Econ 421 – Incentives

Overview
--------
This project studies wage compression among University of California professors
from 2012 to 2024. The central question is whether salary dispersion narrowed
after 2022, when pay transparency became more prominent in California.

The analysis uses publicly available UC salary data, focuses on professor
positions only, and works entirely with nominal wages.

Research Question
-----------------
Do UC professor salaries show evidence of wage compression over time, and
particularly after 2022?

Data
----
Salary data come from two public sources:

1. UC Annual Wage Database (UC Office of the President)
   https://ucannualwage.ucop.edu/wage/
   Used for 2010–2011 (available in professors/ but excluded from the main
   analysis window of 2012–2024).

2. California State Controller Government Compensation Database
   https://publicpay.ca.gov/
   Used for 2012–2024.

All raw data are stored as yearly CSV files in the professors/ directory and
are filtered to include only professor-titled positions.

Each CSV has the following columns:
  Year, EmployerName, Position, RegularPay, TotalWages

Repo Structure
--------------
main.ipynb
    Primary analysis notebook. Run this end to end.

paper.tex / paper.pdf
    LaTeX source and compiled paper.

professors/
    Yearly UC professor salary CSV files (uc_professors_YYYY.csv).
    Covers 2010–2024 but the notebook uses 2012–2024.

figures/
    Pre-generated publication figures and regression tables.
    These are also produced inline in main.ipynb.

export_figures.py
    Standalone script to regenerate all figures in figures/.
    Requires the same dependencies as main.ipynb plus `morethemes`.
    Not required to run main.ipynb.

How to Run
----------
1. Install Python dependencies:
      pip install pandas numpy matplotlib statsmodels scipy

2. Open main.ipynb in Jupyter and run all cells top to bottom.

All data files are included in the repo. No external downloads or setup
steps are required beyond installing the packages above.

Methods Summary
---------------
- Load and standardize professor salary files for 2012–2024
- Classify professors by rank (Assistant, Associate, Full, Clinical, Research)
- Estimate regressions of log wages with linear year trend, rank, and campus
- Measure within-rank salary dispersion using coefficient of variation
- Compare pre-2022 vs post-2022 variance using Levene's test
- Estimate a difference-in-differences compression test (Rank × Post2022)
- Run robustness checks: VIF, deduplication, within-rank regressions

Authors
-------
Agnibha Bhattacharya, Jackson Wurzer, and Joshua Kenworthy
