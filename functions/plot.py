import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandas.api.types import is_numeric_dtype


def coverage_panel(df, prefixes=("efw", "fiw", "ief", "p5d", "pts", "wgi", "wb", "tgt"), start=1960, end=2024, missing_color="lightgrey", dataset_colors=None):

 # set colors per dataset
 if dataset_colors is None:
  dataset_colors = {"efw": "#1f4b99", "fiw": "#1c7c54", "ief": "#b22222", "p5d": "#6a3d9a", "pts": "#e07b39", "wgi": "#007f7f", "wb": "#7a5230", "tgt": "#000000"}
 years = np.arange(start, end + 1)

 # clean dataframe: enforce nulls from *_f, drop *_f and nulls_*
 df = df.copy()
 f_cols = [c for c in df.columns if c.endswith("_f")]
 for col in f_cols:
  base = col[:-2]
  if base in df.columns:
   df.loc[df[col].fillna(0) != 0, base] = pd.NA
 drop_cols = f_cols + [c for c in df.columns if c.startswith("nulls_")]
 df = df.drop(columns=drop_cols, errors="ignore")

 # build long dataset with has-coverage flags
 frames = []
 for p in prefixes:
  p_cols = [c for c in df.columns if c.startswith(p + "_") and is_numeric_dtype(df[c])]
  if not p_cols: continue
  tmp = df[["country", "year"] + p_cols].copy()
  tmp = tmp[(tmp["year"] >= start) & (tmp["year"] <= end)]
  has_any = tmp[p_cols].notna().any(axis=1).astype(int)
  tmp = tmp.loc[:, ["country", "year"]]
  tmp["has"] = has_any
  tmp = tmp.groupby(["country", "year"], as_index=False)["has"].max()
  tmp["dataset"] = p
  frames.append(tmp)
 if not frames: raise ValueError("No usable columns for given prefixes")
 long = pd.concat(frames, ignore_index=True)

 # order rows: dataset blocks, countries alphabetical
 rows = [] 
 block_bounds = []
 for p in prefixes:
  sub = long[long["dataset"] == p]
  if sub.empty: continue
  countries = sorted(sub["country"].unique())
  start_row = len(rows)
  for cty in countries: rows.append((p, cty))
  end_row = len(rows)
  block_bounds.append((p, start_row, end_row))

 # build rgb array
 n_rows = len(rows) 
 n_cols = len(years)
 rgb_array = np.zeros((n_rows, n_cols, 3), dtype=float)
 grey_rgb = mcolors.to_rgb(missing_color)
 rgb_array[:] = grey_rgb

 # fill in dataset colors
 present = {}
 for (p, cty), grp in long.groupby(["dataset", "country"]):
  present[(p, cty)] = set(grp.loc[grp["has"] == 1, "year"].tolist())
 row_index = {rc: i for i, rc in enumerate(rows)}
 for (p, cty), yr_set in present.items():
  if (p, cty) not in row_index: continue
  r = row_index[(p, cty)]
  color_rgb = mcolors.to_rgb(dataset_colors.get(p, "#000000"))
  mask = np.isin(years, list(yr_set))
  rgb_array[r, mask, :] = color_rgb

 # plot
 fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
 ax.imshow(rgb_array, aspect="auto", interpolation="nearest", origin="upper", extent=[years[0] - 0.5, years[-1] + 0.5, n_rows, 0])
 ax.set_title("Country coverage of at least one variable across datasets, 1960-2025", fontsize=14)

 # y-axis labels per dataset block
 yticks = [] 
 ylabels = []
 for p, start_row, end_row in block_bounds:
  mid = (start_row + end_row) / 2
  yticks.append(mid) 
  ylabels.append(p.upper())
  ax.hlines(end_row, years[0] - 0.5, years[-1] + 0.5, colors="white", linewidth=0.4)
 ax.set_yticks(yticks) 
 ax.set_yticklabels(ylabels)

 # x-axis ticks
 step = 5 if (end - start) <= 80 else 10
 ax.set_xticks(np.arange(start, end + 1, step))

 plt.tight_layout() 
 plt.show()
 return fig, ax


def target_timeseries(df, typ, pct1, pct2):
  # keep valid values
  if typ=='yield':
    df_tgt = df[["year", "tgt_yield"]].dropna()
  elif typ=='spread':
    df_tgt = df.copy()
    df_tgt["tgt_spread"] = df_tgt["tgt_yield"] - df_tgt["year"].map(df_tgt.query("country=='United States'").set_index("year")["tgt_yield"])
    df_tgt = df_tgt[["year", "tgt_spread"]].dropna()

  # compute percentiles per year
  quantiles = df_tgt.groupby("year")[f"tgt_{typ}"].quantile([pct1, 0.5, pct2]).unstack()
  quantiles.columns = ["p10", "median", "p90"]

  # plot
  fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
  ax.plot(quantiles.index, quantiles["median"], label="Median", lw=1, color='red')
  ax.plot(quantiles.index, quantiles["p10"], label=f"{pct1*100:.0f}th percentile", ls="--", color='green')
  ax.plot(quantiles.index, quantiles["p90"], label=f"{pct2*100:.0f}th percentile", ls="--", color='orange')

  if typ =='yield':
    ax.set_title("Gov't bond 10Y yields (%), 1960-2024")
  elif typ =='spread':
    ax.set_title("Gov't bond 10Y spread over UST (%), 1960-2024")
  ax.legend()
  ax.grid(which="both", axis="both", linestyle="-", alpha=0.4)
  ax.set_facecolor("#f9f9f9")
  plt.show()