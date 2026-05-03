# instquality — Institutions and Sovereign Borrowing Costs

**Which specific laws and institutions matter most for how much a country pays to borrow? (As measured by the spread over a US 10-year treasury)**

## Motivation

Many developing countries have dozens of institutions in need of reform, and resources for capacity development are finite. Targeting reform at the laws and institutions that most strongly influence sovereign borrowing costs frees up fiscal space — lower interest payments mean more room for productive investment. But to target effectively, we need to know which institutional features matter, how much, over what horizon, and through what channel.

## What the literature shows

Governance and political-institutional conditions are well-established determinants of sovereign borrowing costs:

- **Governance quality.** Stronger government effectiveness, regulatory quality, and control of corruption are associated with lower sovereign spreads and CDS premia, even after controlling for macroeconomic fundamentals ([Jeanneret 2018](https://www.sciencedirect.com/science/article/abs/pii/S0378426618300736); [Beirne & Fratzscher 2013](https://www.sciencedirect.com/science/article/abs/pii/S0261560612001830)).
- **Political risk.** Greater political stability and stronger constraints on the executive are linked to lower spreads and better credit ratings ([Bekaert et al. 2014](https://www.nber.org/system/files/working_papers/w19786/w19786.pdf); [Aizenman, Hutchison & Jinjarak 2013](https://www.econstor.eu/bitstream/10419/64488/1/647513501.pdf)).
- **Fiscal institutions and transparency.** More transparent, rules-based budget processes and higher fiscal transparency are associated with lower borrowing costs and improved ratings ([Hameed 2005](https://www.imf.org/external/pubs/ft/wp/2005/wp05225.pdf); [Glennerster & Shin 2003](https://www.imf.org/external/pubs/ft/wp/2003/wp03132.pdf); [International Budget Partnership 2022](https://internationalbudget.org/open-budget-survey/)).

The literature, however, typically relies on broad composite indices that obscure which components drive the effect, over what horizon, and through what channel.

## Contribution

This project moves from *"rule of law matters"* to *"rule of law matters **more than** control of corruption"* by:

1. **Granular variables.** A horse race across **100+ institutional and macro variables** from seven sources, rather than broad composite indices.
2. **Two channels.** Decomposing institutional effects into (i) *structural priced-in levels* — what determines long-run spread levels in the cross-section — and (ii) *incremental predictive content* — what actually moves spreads when fundamentals change.
3. **Horizons.** Tracing both effects across 1–10 year horizons to show how institutional influence evolves over time — a dimension largely absent from existing work.

## Central findings

1. **Structural legal quality is the dominant long-run determinant of borrowing costs.** Judicial independence and integrity of the legal system overtake all other institutional factors at horizons of 5–10 years.
2. **Short-run spread movements are almost entirely persistence.** Institutional and macro fundamentals add no marginal predictive signal at 1–3 year horizons.
3. **Medium-to-long-run spread *changes* respond to openness, not core legal institutions.** Slow-moving legal quality sets the level; openness (e.g. *Freedom of Foreigners to Visit*) drives the dynamics.

For current pricing in the cross-section, the three institutional variables explaining most of the variation are **Monetary Freedom**, the **(time) Cost of Importing and Exporting**, and **Integrity of the Legal System**.

---

## Data

Variables drawn from seven sources, covering 1960–2023:

| Source | Description | Variables |
|---|---|---|
| EFW (Fraser Institute) | Economic Freedom of the World | 79 |
| FIW (Freedom House) | Freedom in the World | 16 |
| IEF (Heritage Foundation) | Index of Economic Freedom | 13 |
| Polity V | Political regime characteristics | 17 |
| PTS | Political Terror Scale | 6 |
| WGI (World Bank) | World Governance Indicators | 36 |
| WB | World Bank macro + institutional | 46 |

**Dependent variable:** 10-year sovereign spread over US Treasuries, constructed from OECD and World Bank bond yield data (1960–2024).

---

## Methods

Two complementary regression frameworks, both implemented with **LASSO, Ridge, and Elastic Net** and expanding-window cross-validation for time-series consistency:

1. **Incremental signal analysis** — regresses the future spread on its current value plus institutional fundamentals. The autoregressive term absorbs persistence (i.e. that high-spread countries tend to stay high-spread), so the coefficients on fundamentals capture only what they add *beyond* what the current spread already tells us. Question answered: *do institutions help forecast where spreads are headed, on top of where they already are?*

2. **Priced-in levels analysis** — regresses the spread itself on institutional fundamentals, with no autoregressive control. Coefficients capture the structural, cross-sectional relationship between institutional quality and the level of borrowing costs. Question answered: *how much of why Argentina pays more than Germany is explained by institutions?*

The distinction matters because the two can — and in this case do — give very different answers. Levels are heavily institutional (priced-in), but year-on-year *movements* are dominated by persistence, with institutions adding marginal signal only at longer horizons.

Both frameworks are extended across 1–10 year horizons to capture evolving dynamics.

Feature preparation includes imputation, scaling, hierarchical clustering to remove redundant correlated variables, and SHAP-based feature attribution aggregated by core economic indicator. Stability selection identifies which variables are chosen consistently across subsamples. Model accuracy is benchmarked using R², RMSE, and Diebold-Mariano tests.

---

## Structure

```
instquality/
├── 01_load_and_visualise.ipynb    # Data overview, distributions, correlations
├── 02_preprocess.ipynb            # Cleaning, missing value handling, feature engineering
├── 03_run_base_analysis.ipynb     # Incremental signal and priced-in levels (baseline)
├── 04_run_temporal_analysis.ipynb # Horizon-by-horizon regressions (1–10 years)
├── 05_summary.ipynb               # Full results summary (rendered: see link below)
├── functions/                     # Reusable regression framework and labels
└── files/                         # Data inputs
```

Rendered summary: [janjdarecki.github.io/instquality/05_summary.html](https://janjdarecki.github.io/instquality/05_summary.html)
