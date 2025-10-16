import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator


def coverage_panel(df, prefixes=("efw", "fiw", "ief", "p5d", "pts", "wgi", "wb", "tgt"), start=1960, end=2024, missing_color="#f5f5f5"):
 dataset_colors = {
      "efw": "purple",
      "fiw": "#10b981",
      "ief": "#ef4444",
      "p5d": "maroon",
      "pts": "#f59e0b",
      "wgi": "#06b6d4",
      "wb":  "#3b82f6",
      "tgt": "#111827"
  }
 years = np.arange(start, end + 1)
 df = df.copy()
 
 f_cols = [c for c in df.columns if c.endswith("_f")]
 for col in f_cols:
  base = col[:-2]
  if base in df.columns:
   df.loc[df[col].fillna(0) != 0, base] = pd.NA
 drop_cols = f_cols + [c for c in df.columns if c.startswith("nulls_")]
 df = df.drop(columns=drop_cols, errors="ignore")
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
 n_rows = len(rows) 
 n_cols = len(years)
 rgb_array = np.zeros((n_rows, n_cols, 3), dtype=float)
 grey_rgb = mcolors.to_rgb(missing_color)
 rgb_array[:] = grey_rgb

 present = {}
 for (p, cty), grp in long.groupby(["dataset", "country"]):
  present[(p, cty)] = set(grp.loc[grp["has"] == 1, "year"].tolist())
 row_index = {rc: i for i, rc in enumerate(rows)}
 
 for (p, cty), yr_set in present.items():
  if (p, cty) not in row_index: continue
  r = row_index[(p, cty)]
  color_rgb = mcolors.to_rgb(dataset_colors.get(p, "#f5f5f5"))
  mask = np.isin(years, list(yr_set))
  rgb_array[r, mask, :] = color_rgb
 fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
 ax.imshow(rgb_array, aspect="auto", interpolation="nearest", origin="upper", extent=[years[0] - 0.5, years[-1] + 0.5, n_rows, 0])
 ax.set_title("Country coverage of at least one variable across datasets, 1960-2025", fontsize=14)
 yticks = [] 
 ylabels = []
 
 for p, start_row, end_row in block_bounds:
  mid = (start_row + end_row) / 2
  yticks.append(mid) 
  ylabels.append(p.upper())
  ax.hlines(end_row, years[0] - 0.5, years[-1] + 0.5, colors="white", linewidth=0.4)
 
 for side in ("top", "right", "left", "bottom"):
  ax.spines[side].set_alpha(0)
 ax.set_yticks(yticks) 
 ax.set_yticklabels(ylabels)
 step = 5 if (end - start) <= 80 else 10
 ax.set_xticks(np.arange(start, end + 1, step))
 plt.tight_layout() 
 plt.show()
 return fig, ax


def coverage_per_variable(df):
    exclude_cols = ["country", "year", "region", "iso_code_1", "iso_code_2"]
    prefixes = ["efw", "fiw", "ief", "p5d", "pts", "wgi", "wb", "tgt"]
    dataset_colors = {
        "efw": "purple",
        "fiw": "#10b981",
        "ief": "#ef4444",
        "p5d": "maroon",
        "pts": "#f59e0b",
        "wgi": "#06b6d4",
        "wb":  "#3b82f6",
        "tgt": "#111827"
    }
    label_map = {
        "efw": "Economic Freedom of the World",
        "fiw": "Freedom in the World",
        "ief": "Index of Economic Freedom",
        "p5d": "Polity5D Project",
        "pts": "Political Terror Scale",
        "wgi": "Worldwide Governance Indicators",
        "wb":  "World Bank (macro)",
        "tgt": "10Y Gov't yield (target)"
    }

    cols = [c for c in df.columns if c not in exclude_cols]
    coverage = (df[cols].notna().mean() * 100).to_dict()
    cov_by_prefix = {p: [] for p in prefixes}
    for col, cov in coverage.items():
        pref = next((p for p in prefixes if col.startswith(p + "_")), None)
        if pref is not None:
            cov_by_prefix[pref].append(cov)
    data   = [cov_by_prefix[p] for p in prefixes if cov_by_prefix[p]]
    colors = [dataset_colors[p]  for p in prefixes if cov_by_prefix[p]]
    labels = [label_map[p]       for p in prefixes if cov_by_prefix[p]]
    if not data:
        print("No columns matched the specified prefixes.")
        return

    bins = np.arange(0, 102, 5)
    plt.figure(figsize=(7.5, 4.5), dpi=125)
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")  # light grey INSIDE the plot
    n, b, patches = plt.hist(
        data,
        bins=bins,
        stacked=True,
        color=colors,
        label=labels
    )
    for pgroup in patches:
        for p in pgroup:
            p.set_linewidth(0.8)
            p.set_edgecolor("#ffffff")
            p.set_alpha(0.95)

    ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.5)
    ax.grid(axis="x", which="both",  linewidth=0.4, alpha=0.25)
    plt.xlabel("Coverage (%)")
    plt.ylabel("Number of variables")
    plt.title(f"Per-variable coverage distribution by dataset - {df.country.nunique()} countries, 1960-2024")
    plt.xticks(np.arange(0, 101, 20), [f"{v}" for v in np.arange(0, 101, 20)])
    plt.legend(title="", frameon=True, ncols=1, loc='upper right', fontsize=11)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_alpha(0)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()


def coverage_per_version(df):
    remove = (df.groupby('country')['tgt_yield'].apply(lambda x: x.notna().any()).loc[lambda s: ~s].index)
    df_tgt = df[~df['country'].isin(remove)].copy()
    df_res = df_tgt[df_tgt.year>=1990]
    exclude_cols = ["country", "year", "region", "iso_code_1", "iso_code_2"]
    cols1 = [c for c in df.columns if c not in exclude_cols]
    cols2 = [c for c in df_tgt.columns if c not in exclude_cols]
    cols3 = [c for c in df_res.columns if c not in exclude_cols]
    common_cols = sorted(set(cols1) & set(cols2) & set(cols3))
    if not common_cols:
        print("No common variables to compare after excluding metadata columns.")
        return
    cov_df   = (df[common_cols].notna().mean() * 100).values
    cov_tgt  = (df_tgt[common_cols].notna().mean() * 100).values
    cov_res  = (df_res[common_cols].notna().mean() * 100).values

    bins = np.arange(0, 102, 2.5)
    plt.figure(figsize=(7.5, 4.5), dpi=125)
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")  # light grey INSIDE the plot
    c_df, c_tgt, c_res = "#3b82f6", "#10b981", "#f59e0b"  # blue / green / amber

    def draw_hist(data, color, label):
        ax.hist(data, bins=bins, alpha=0.28, color=color, label=label, histtype="stepfilled")
        ax.hist(data, bins=bins, histtype="step", linewidth=1.6, color=color)
    draw_hist(cov_df,  c_df,  "Starting dataframe")
    draw_hist(cov_tgt, c_tgt, "Restricted to countries\nwith target data")
    draw_hist(cov_res, c_res, "With additional\nrestriction from 1990")

    ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.5)
    ax.grid(axis="x", which="both",  linewidth=0.4, alpha=0.25)
    plt.xlabel("Coverage (%)")
    plt.ylabel("Number of variables")
    plt.title("Per-variable coverage distribution (overlay across data versions)")
    plt.xticks(np.arange(0, 101, 20), [f"{v}" for v in np.arange(0, 101, 20)])
    plt.legend(fontsize=10, frameon=True, ncols=3, bbox_to_anchor=(0.99,1))
    plt.ylim(0,68)
    plt.xlim(0,100)
    for side in ("top", "right", "left", "bottom"):
      ax.spines[side].set_alpha(0)
    plt.tight_layout()
    plt.show()


def target_coverage(df, excl_missing=False):
  df2 = df.copy()
  if excl_missing:
    df2 = df2[df2.groupby('country')['tgt_yield'].transform('count') > 0]
  df2['year'] = pd.to_numeric(df2['year'], errors='coerce')
  df2 = df2[df2['year'].between(1960, 2024)]
  df2['has'] = df2['tgt_yield'].notna().astype(int)
  mask = df2.pivot_table(index='country', columns='year', values='has',
                        aggfunc='max', fill_value=0).astype(bool)
  mask.columns = mask.columns.astype(int)
  if mask.columns.duplicated().any():
      mask = mask.groupby(mask.columns, axis=1).max()
  mask = mask.sort_index(axis=0)  # alphabetical countries
  mask = mask.sort_index(axis=1)  # ascending years

  fig, ax = plt.subplots(figsize=(18, max(8, len(mask)/4)))  # auto-scale height
  sns.heatmap(mask, cmap=["lightgrey", "darkblue"], cbar=False, ax=ax)
  ax.set_title("Data availability by country and year (10Y yield)")
  ax.set_xlabel("Year")
  ax.set_ylabel("Country")
  has_data = mpatches.Patch(color="darkblue", label="Has data")
  missing  = mpatches.Patch(color="lightgrey", label="Missing")
  ax.legend(handles=[has_data, missing], loc="upper right", bbox_to_anchor=(1.15, 1))
  plt.show()
  coverage = mask.sum(axis=0)  # number of countries with data each year
  fig, ax = plt.subplots(figsize=(12, 4))
  ax.plot(coverage.index, coverage.values*100/df2.country.nunique(), lw=2)
  ax.set_title("Percent of countries with data by year (%)")
  ax.set_xlabel("Year")
  plt.ylim(0,100)
  plt.show()


def vars_timeseries(df, label_map, countries, colors):
    df_plot = df[df["country"].isin(countries)].copy()
    df_plot["year"] = df_plot["year"].astype(int)
    
    for variable in [v for v in label_map.keys() if v in df.columns]:
        label = label_map.get(variable, variable)
        df_var = df_plot[df_plot[variable].notnull()].copy()
        if df_var.empty:
            continue
        xmin, xmax = df_var["year"].min(), df_var["year"].max()
        plt.figure(figsize=(8, 4.5), dpi=125)
        ax = plt.gca()
        ax.set_facecolor("#f5f5f5")
        
        for c, col in zip(countries, colors):
            data_country = df_var[df_var["country"] == c]
            if data_country.empty:
                continue
            plt.plot(
                data_country["year"],
                data_country[variable],
                label=c,
                linewidth=2.2,
                color=col
            )
        
        plt.title(label, fontsize=12.5)
        plt.grid(alpha=0.4, linewidth=0.6)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_alpha(0)
        plt.legend(frameon=False, bbox_to_anchor=(1, 1.025), fontsize=11)
        ax.set_xlim(xmin-2, xmax+2)
        ax.xaxis.set_major_locator(MultipleLocator(10))   # major tick every 5 years
        # ax.xaxis.set_minor_locator(MultipleLocator(5))   # minor tick every 1 year
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


# target

def target_timeseries(df, typ, pct1, pct2):
  if typ=='yield':
    df_tgt = df[["year", "tgt_yield"]].dropna()
  elif typ=='spread':
    df_tgt = df.copy()
    df_tgt["tgt_spread"] = df_tgt["tgt_yield"] - df_tgt["year"].map(df_tgt.query("country=='United States'").set_index("year")["tgt_yield"])
    df_tgt = df_tgt[["year", "tgt_spread"]].dropna()
  quantiles = df_tgt.groupby("year")[f"tgt_{typ}"].quantile([pct1, 0.5, pct2]).unstack()
  quantiles.columns = ["p10", "median", "p90"]

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


def compare_target_sources(df, mode="both"):
    g = df.groupby("country")[["oecd_yield_long", "imf_yield_long"]].apply(
        lambda x: pd.Series({
            "oecd_n": x["oecd_yield_long"].notnull().sum(),
            "imf_n":  x["imf_yield_long"].notnull().sum()}))
    if mode == "both":
        countries = g[(g["oecd_n"] >= 2) & (g["imf_n"] >= 2)].index
    elif mode == "oecd":
        countries = g[(g["oecd_n"] >= 2) & (g["imf_n"] < 2)].index
    elif mode == "imf":
        countries = g[(g["imf_n"] >= 2) & (g["oecd_n"] < 2)].index
    else:  # "either"
        countries = g[(g["oecd_n"] >= 2) | (g["imf_n"] >= 2)].index

    for country in sorted(countries):
        cntry = df[df.country == country]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        if cntry["oecd_yield_long"].notnull().any():
            ax.plot(cntry["year"], cntry["oecd_yield_long"], lw=1, color="red", label="OECD")
        if cntry["imf_yield_long"].notnull().any():
            ax.plot(cntry["year"], cntry["imf_yield_long"], lw=1, color="blue", label="IMF")
        ax.set_title(f"{country} Gov't bond 10Y yields (%), 1960–2024")
        ax.legend(title="Source")
        ax.grid(which="both", axis="both", linestyle="-", alpha=0.4)
        ax.set_facecolor("#f9f9f9")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()


def region_violin(df):
  region_map = {
      "Southern Europe": "Europe",
      "Eastern Europe": "Europe",
      "Western Europe": "Europe",
      "Northern Europe": "Europe",
      "Central Asia": "Asia",
      "Eastern Asia": "Asia",
      "South-eastern Asia": "Asia",
      "Southern Asia": "Asia",
      "Western Asia": "Asia",
      "Northern Africa": "Africa",
      "Western Africa": "Africa",
      "Eastern Africa": "Africa",
      "Middle Africa": "Africa",
      "Southern Africa": "Africa",
      "Northern America": "Americas",
      "Central America": "Americas",
      "South America": "Americas",
      "Caribbean": "Americas",
      "Australia and New Zealand": "Australia & Oceania",
      "Melanesia": "Australia & Oceania",
      "Polynesia": "Australia & Oceania"
  }

  df["continent"] = df["region"].map(region_map)
  df_group = (df.groupby(["country", "continent"], as_index=False)["tgt_yield"].mean())
  df_group = df_group[df_group.tgt_yield.notnull()]
  counts = (df_group.groupby("continent")["country"].nunique().rename("n_countries"))
  df_group = df_group.merge(counts, on="continent", how="left")
  df_group["continent_label"] = df_group.apply(lambda x: f"{x['continent']} ({x['n_countries']} countr{'ies' if x['n_countries']>1 else 'y'})", axis=1)

  figsize = (12, 1.2 * df_group["continent"].nunique())
  plt.figure(figsize=figsize)
  sns.violinplot(
      data=df_group,
      x="tgt_yield",
      y="continent_label",
      inner="stick",      # shows each country's line
      scale="width",
      palette="Dark2",
      cut=0)

  sns.despine(top=True, right=True, bottom=True, left=True)
  plt.title("Average 10Y Yield per Country, by Region (%)")
  plt.xlabel("")
  plt.tight_layout()
  plt.ylabel("")
  plt.grid()
  plt.axvline(x=0, color="red", linestyle="-")
  plt.show()