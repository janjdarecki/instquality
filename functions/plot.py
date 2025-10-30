import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
from pandas.api.types import is_numeric_dtype
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


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
        "wb":  "World Bank",
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
    plt.legend(title="", frameon=True, ncols=1,
     #loc='upper right',
      fontsize=11)
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


def vars_timeseries(df, label_map, countries, colors):
    df_plot = df[df["country"].isin(countries)].copy()
    df_plot["year"] = df_plot["year"].astype(int)

    for variable in [v for v in label_map.keys() if v in df.columns]:
        label = label_map.get(variable, variable)
        df_var = df_plot[df_plot[variable].notnull()].copy()
        if df_var.empty:
            continue

        # Infer dataset prefix (e.g. "efw" → "[EFW]")
        prefix = variable.split("_")[0].upper()
        title_prefix = f"[{prefix}] "

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

        plt.title(f"{title_prefix}{label}", fontsize=12.5)
        plt.grid(alpha=0.4, linewidth=0.6)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_alpha(0)
        plt.legend(frameon=False, bbox_to_anchor=(1, 1.02), loc='upper left', fontsize=11)
        ax.set_xlim(xmin - 2, xmax + 2)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


##### target #######

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


def target_coverage_start(df, cumulative=False, color="#4a90e2"):
    dfn = df.dropna(subset=["tgt_yield"])[["country", "year"]].copy()
    dfn["year"] = dfn["year"].astype(int)
    first_years = dfn.groupby("country", as_index=False)["year"].min().rename(columns={"year": "first_year"})
    first_counts = first_years["first_year"].value_counts().sort_index()
    active_counts = dfn.groupby("year")["country"].nunique().sort_index()
    year_index = pd.Index(range(1960, 2025), name="year")
    counts = (active_counts if cumulative else first_counts).reindex(year_index, fill_value=0).astype(int)

    plt.figure(figsize=(6, 4), dpi=125)
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")
    ax.bar(counts.index, counts.values, color=color, edgecolor="none", width=0.8)
    plt.title("Number of countries with target data per year" if cumulative else "Number of countries by first year of available target data", fontsize=12.5)
    plt.grid(alpha=0.4, linewidth=0.6, axis="y", linestyle='-')
    plt.grid(alpha=0.4, linewidth=0.6, axis="x", linestyle='-')
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_alpha(0)
    ax.set_xlim(1959, 2025)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()


def target_timeseries(df, typ, pct1, pct2):
  if typ=='yield':
    df_tgt = df[["year", "tgt_yield"]].dropna()
  elif typ=='spread':
    df_tgt = df.copy()
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


def target_histogram(df, variable="tgt_yield", country_average=True):
    df = df.copy()
    df["continent"] = df["region"].map(region_map)
    df = df[df[variable].notnull()]
    if country_average:
        df = df.groupby(["country", "continent"], as_index=False)[variable].mean()
        n_bins = 25
        x_min, x_max = [None, None]
        opt = "country average "
    else:
        n_bins = 250
        opt = ""
        if variable == 'tgt_spread':
          x_min, x_max = [-7, 22]
        else:
          x_min, x_max = [-1.5, 27]

    continents = df["continent"].dropna().unique()
    palette = sns.color_palette("Dark2", n_colors=len(continents))
    color_map = dict(zip(sorted(continents), palette))
    data_by_continent = [df.loc[df["continent"] == c, variable] for c in sorted(continents)]
    plt.figure(figsize=(6.5, 4), dpi=125)
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")
    
    bins = np.histogram_bin_edges(df[variable], bins=n_bins)
    plt.hist(data_by_continent,
             bins=bins,
             stacked=True,
             color=[color_map[c] for c in sorted(continents)],
             label=sorted(continents),
             alpha=0.9)
    plt.title(f"Distribution of {opt}{variable.replace('tgt_', '')} values", fontsize=12.5)
    plt.xlabel(f"{variable.replace('tgt_', '').capitalize()} (%)")
    plt.ylabel("Count")
    plt.grid(alpha=0.4, linewidth=0.6)
    if x_max is not None:
      plt.xlim(x_min, x_max)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_alpha(0)
    plt.legend(title="", frameon=True, fontsize=11, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def target_timeseries_per_var_quartile(df, target_col, totals, colors=None):
    dfp = df.copy()
    dfp["year"] = dfp["year"].astype(int)
    variables = [k for k in totals.keys() if k in dfp.columns]
    if colors is None:
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    title_metric = {"tgt_spread": "Mean 10-Year Sovereign Spread over U.S. Treasuries (%)",
                    "tgt_yield": "Mean 10-Year Government Bond Yield (%)"}.get(target_col, f"Mean {target_col}")
    legend_labels = {4: "Best quartile", 3: "Second best", 2: "Second worst", 1: "Worst quartile"}

    for var in variables:
        dfv = dfp if var != "p5d_polity" else dfp[dfp["year"] <= 2018]
        d = dfv[["country","year",var,target_col]].dropna(subset=[var,target_col]).copy()
        if d.empty:
            continue

        def _quartiles_year(g):
            s = g[var].astype(float)
            if s.nunique() < 4:
                r = s.rank(pct=True, method="average")
                return pd.cut(r, bins=[0,.25,.5,.75,1], include_lowest=True, labels=[1,2,3,4]).astype("Int64")
            try:
                return pd.qcut(s, 4, labels=[1,2,3,4]).astype("Int64")
            except Exception:
                r = s.rank(pct=True, method="average")
                return pd.cut(r, bins=[0,.25,.5,.75,1], include_lowest=True, labels=[1,2,3,4]).astype("Int64")

        d["quartile"] = d.groupby("year", group_keys=False).apply(_quartiles_year)
        g = d.dropna(subset=["quartile"]).groupby(["year","quartile"])[target_col].mean().reset_index()
        if g.empty:
            continue
        p = g.pivot(index="year", columns="quartile", values=target_col).sort_index()
        if p.empty or pd.isna(p.index.min()):
            continue

        xmin, xmax = int(p.index.min()), int(p.index.max())
        plt.figure(figsize=(8, 4.5), dpi=125)
        ax = plt.gca(); ax.set_facecolor("#f5f5f5")
        for q, col in zip([4,3,2,1], colors):
            if q in p.columns:
                plt.plot(p.index, p[q], label=legend_labels[q], linewidth=2.2, color=col)

        prefix = var.split("_")[0].upper()
        title_prefix = f"[{prefix}] "
        var_label = totals.get(var, var)
        plt.title(f"{title_metric} by\n{title_prefix}{var_label} quartile", fontsize=12.5)
        plt.grid(alpha=0.4, linewidth=0.6)
        for side in ("top","right","left","bottom"):
            ax.spines[side].set_alpha(0)
        plt.legend(frameon=False, bbox_to_anchor=(1, 1.02), loc="upper left", fontsize=11)
        ax.set_xlim(xmin - 2, xmax + 2)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


#### correlations #####

def correlation_across_datasets(df,label_dicts,method="pearson"):
    groups={k:[c for c in v.keys() if c in df.columns and is_numeric_dtype(df[c])] for k,v in label_dicts.items()}
    groups={k:[c for c in cols if df[c].notna().sum()>2] for k,cols in groups.items()}
    groups={k:cols for k,cols in groups.items() if cols}
    keys=list(groups.keys())
    labels=[k.upper() if k!="controls" else "Controls" for k in keys]
    M=pd.DataFrame(index=labels,columns=labels,dtype=float)
    for (k1,l1) in zip(keys,labels):
        cols1=groups[k1]
        for (k2,l2) in zip(keys,labels):
            cols2=groups[k2]
            C=df[cols1+cols2].corr(method=method,min_periods=3)
            if k1==k2:
                a=C.loc[cols1,cols1].abs().values
                v=np.nanmean(a[np.triu_indices_from(a,1)])
            else:
                v=np.nanmean(C.loc[cols1,cols2].abs().values)
            M.loc[l1,l2]=v
    fig,ax=plt.subplots(figsize=(8.6,7.0),dpi=125)
    cmap="coolwarm"; vmin,vmax=0,1
    sns.heatmap(M.astype(float),annot=True,fmt=".2f",vmin=vmin,vmax=vmax,cmap=cmap,square=True,linewidths=.7,linecolor="#f5f5f5",cbar=False,mask=M.isna(),annot_kws={"fontsize":9})
    ax.set_title("Average absolute correlations between datasets",pad=8,fontsize=12.3,fontweight="regular")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    for t in ax.get_xticklabels()+ax.get_yticklabels():
        t.set_color("black"); t.set_fontsize(10)
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(False)
    norm=mcolors.Normalize(vmin=vmin,vmax=vmax); smap=plt.cm.get_cmap(cmap)
    for txt in ax.texts:
        try: val=float(txt.get_text())
        except ValueError: continue
        r,g,b,_=smap(norm(val)); lum=0.2126*r+0.7152*g+0.0722*b
        txt.set_color("black" if lum>0.6 else "white"); txt.set_fontweight("medium")
    n=len(M)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i,i),1,1,fill=False,ec="#dddddd",lw=0.6,alpha=0.7))
    ax.tick_params(length=0,pad=4)
    plt.tight_layout(); plt.show()
    return M


def top_correlations(df,label_dicts,target="tgt_spread",top_n=50):
    all_labels={k:v for d in label_dicts.values() for k,v in d.items()}
    vars_=[c for c in all_labels if c in df.columns and is_numeric_dtype(df[c]) and c!=target]
    sub=df[[target]+vars_].dropna(subset=[target])
    corrs=sub.corr(method="pearson")[target].abs().sort_values(ascending=False)
    corrs=corrs.drop(target,errors="ignore").head(top_n)
    formatted=[]
    for v in corrs.index:
        if v.startswith("wb_iq_"):
            prefix = "WB"
        elif v.startswith("wb_"):
            prefix = "CTRL"
        else:
            prefix = v.split("_")[0].upper()
        name = all_labels.get(v, v)
        formatted.append(f"[{prefix}] {name}")
    base_height=10
    base_n=40
    fig_height=base_height*(top_n/base_n)
    fig,ax=plt.subplots(figsize=(8.6,fig_height),dpi=125)
    sns.barplot(x=corrs.values,y=formatted,orient="h",palette="crest",ax=ax)
    for i,v in enumerate(corrs.values):
        ax.text(v+0.01,i,f"{v:.2f}",va="center",fontsize=8)
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} variables with highest absolute Pearson correlation vs. 10-year sovereign spread over U.S. treasuries",
                 pad=12.5, x=-0.1, fontsize=15,fontweight="regular")
    ax.grid(True,axis="x",alpha=0.4)
    sns.despine(left=True,bottom=True)
    plt.tight_layout()
    plt.show()
    return corrs


def correlation_across_top_vars(df,label_dicts,target="tgt_spread",top_n=10,exclude_ctrl=False):
    all_labels={k:v for d in label_dicts.values() for k,v in d.items()}
    def _is_ctrl(v): return v.startswith("wb_") and not v.startswith("wb_iq_")
    vars_=[c for c in all_labels if c in df.columns and is_numeric_dtype(df[c]) and c!=target]
    if exclude_ctrl: vars_=[v for v in vars_ if not _is_ctrl(v)]
    sub=df[[target]+vars_].dropna(subset=[target])
    ranked=sub.corr(method="pearson")[target].abs().sort_values(ascending=False)
    ranked=ranked.drop(target,errors="ignore")
    top=ranked.head(top_n).index.tolist()
    corr=sub[top].corr(method="pearson").abs()
    def _pref(v):
        if v.startswith("wb_iq_"): return "WB"
        elif v.startswith("wb_"):  return "CTRL"
        else:                      return v.split("_")[0].upper()
    labels=[f"[{_pref(v)}] {all_labels.get(v,v)}" for v in corr.columns]
    fig,ax=plt.subplots(figsize=(top_n+4,top_n-1),dpi=125)
    sns.heatmap(corr,vmin=0,vmax=1,cmap="coolwarm",square=True,annot=True,fmt=".2f",
                linewidths=.7,linecolor="#f0f0f0",cbar=False)
    ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=12)
    ax.set_yticklabels(labels,rotation=0,fontsize=13.5)
    title=f"Correlation heatmap across top {top_n} variables with highest absolute Pearson\ncorrelations with 10-year sovereign bond spread"
    ax.set_title(title,fontsize=15,fontweight="regular",pad=15)
    plt.tight_layout(); plt.show()
    return corr


def dendrogram_for_top_vars(corr, label_dicts, method="average"):
    A=corr.abs().copy(); np.fill_diagonal(A.values,1); dist=1-A
    Z=linkage(squareform(dist,checks=False),method=method)
    all_labels={k:v for d in label_dicts.values() for k,v in d.items()}
    def _pref(v):
        if v.startswith("wb_iq_"): return "WB"
        elif v.startswith("wb_"):  return "CTRL"
        else: return v.split("_")[0].upper()
    labels=[f"[{_pref(c)}] {all_labels.get(c,c)}" for c in A.columns]
    fig,ax=plt.subplots(figsize=(9,5.5),dpi=125,facecolor="white")
    ax.set_facecolor("#f5f5f5")
    dendrogram(Z,labels=labels,orientation="right",leaf_font_size=9,
               color_threshold=0.7,above_threshold_color="#444444",ax=ax)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    ax.set_xlabel("1 − absolute correlation",fontsize=9,labelpad=8)
    ax.tick_params(axis="x",labelsize=8)
    ax.set_title("Hierarchical clustering of top 15 institutional variables with highest absolute Pearson correlations with target variable",
                 fontsize=11,x=0.1,fontweight="regular",pad=10)
    plt.tight_layout()
    plt.show()


#### methods #####

def coverage_progression(df):
  dff = df.dropna(subset=["tgt_spread_lag"]).copy()
  dff = dff.sort_values(["country", "year"])
  dff["cum_share"] = (dff.groupby("country").cumcount() + 1) / dff.groupby("country")["year"].transform("count")
  cross_years = dff.loc[dff["cum_share"] >= 0.75].groupby("country")["year"].min()
  split_year = int(cross_years.median())
  avg_split_year = int(cross_years.mean())

  plt.figure(figsize=(6.5,4), dpi=125)
  ax = plt.gca()
  ax.set_facecolor("#f5f5f5")
  for _, g in dff.groupby("country"):
      plt.plot(g["year"], g["cum_share"] * 100, lw=0.9, alpha=0.55)
  plt.axhline(75, color="#d62728", ls="--", lw=1)
  plt.title("Per-country progression of coverage (%)", pad=10)
  for spine in ["top", "left", "bottom", "right"]:
      ax.spines[spine].set_visible(False)
  ax.spines["left"].set_alpha(0.4)
  ax.spines["bottom"].set_alpha(0.4)
  ax.grid(False)
  ax.patch.set_alpha(1)
  plt.ylim(0, 102)
  plt.tight_layout()
  plt.show()



#### region map #####

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