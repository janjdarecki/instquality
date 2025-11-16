import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from functions.utils import check_split_year


# mean reversion benchmark


def show_simple_benchmark(df, split_share):
    # map split_share to label
    split_labels = {0.75: "75‑25", 0.80: "80‑20", 0.85: "85‑15"}
    split_label = split_labels.get(split_share, f"{int(split_share*100)}‑{int((1-split_share)*100)}")
    
    # prep
    full = df[["country", "year", "tgt_spread_t1", "tgt_spread"]].dropna()
    fullte = full[full["year"] > check_split_year(df, "tgt_spread_t1", split_share=split_share)]
    y_true = fullte["tgt_spread_t1"]
    y_pred = fullte["tgt_spread"]
    
    # get main stats
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    
    # show fit and prediction error
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=20)
    axes[0].plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                color="red", linestyle="-", lw=1)
    axes[0].set_xlabel("Actual spread (t+1)")
    axes[0].set_ylabel("Predicted spread (t)")
    axes[0].set_title(f"Mean-reversion benchmark ({split_label})\nRMSE={rmse:.2f}, R²={r2:.3f}")
    
    errors = y_true - y_pred
    axes[1].hist(errors, bins=40, color="steelblue", alpha=0.7)
    axes[1].axvline(0, color="red", lw=1)
    axes[1].set_title("Forecast errors")
    axes[1].set_xlabel("Error (actual − predicted)")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.show()
    
    # how much of total variance is explained by between-country variance
    df = fullte[["country", "tgt_spread_t1"]].dropna()
    overall_mean = df["tgt_spread_t1"].mean()
    n_i = df.groupby("country")["tgt_spread_t1"].count()
    means_i = df.groupby("country")["tgt_spread_t1"].mean()
    
    # weighted between-country variance
    sst_between = ((n_i * (means_i - overall_mean)**2).sum()) / (len(df) - 1)
    sst_total = ((df["tgt_spread_t1"] - overall_mean)**2).sum() / (len(df) - 1)
    print(f"Share of between-country variance in total variance: {sst_between / sst_total:.2%}")


# base sigal regressions


def get_label(var_name, all_labels):
    return all_labels.get(var_name, var_name)


def simplify_label(label):
    base = re.sub(r'\(.*?\)', '', label)
    base = base.split('.')[-1].strip()
    return base


def summarise_signal_regressions(specs_dir, base_name, models, benchmark_dfs):
    for split_share, (split_tag, split_label) in {
        0.75: ("", "75-25"),
        0.80: ("_80", "80-20"),
        0.85: ("_85", "85-15")
    }.items():
        summary_data = []
        benchmark_df = benchmark_dfs.get(split_share)
        if benchmark_df is None or benchmark_df.empty:
            continue
        bench_r2 = benchmark_df.loc[benchmark_df["horizon"] == 1, "R2_test"].values[0]
        bench_rmse = benchmark_df.loc[benchmark_df["horizon"] == 1, "RMSE_test"].values[0]

        for model in models:
            for l1_ratio in ([0.25, 0.5, 0.75] if model == "elastic" else [None]):
                file_name = (
                    f"{model}_{base_name}_t1{split_tag}_l1_{l1_ratio}_clust_results.dat"
                    if model == "elastic"
                    else f"{model}_{base_name}_t1{split_tag}_clust_results.dat"
                )
                path = os.path.join(specs_dir, file_name)
                if not os.path.exists(path):
                    continue
                df = pd.read_pickle(path)
                if df.empty or "R²_test" not in df:
                    continue
                best = df.loc[df["R²_test"].idxmax()]
                reject = "Accepted" if (pd.notna(best["DM_p"]) and best["DM_p"] < 0.05) else "Rejected"
                label = f"{model.capitalize()}" if l1_ratio is None else f"Elastic Net (L1={l1_ratio})"
                summary_data.append({
                    "Model": label,
                    "R²_test": best["R²_test"],
                    "RMSE_test": best["RMSE_test"],
                    "ΔR² vs Benchmark (p.p.)": (best["R²_test"] - bench_r2) * 100,
                    "ΔRMSE vs Benchmark (bps)": (best["RMSE_test"] - bench_rmse) * 100,
                    "H1=Incremental signal present": reject,
                    "DM_stat": best["DM_stat"] if pd.notna(best["DM_stat"]) else None,
                    "DM_p": best["DM_p"] if pd.notna(best["DM_p"]) else None,
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values("Model", ascending=False)
            df["R²_test"] = df["R²_test"].map(lambda x: f"{x:.4f}")
            df["RMSE_test"] = df["RMSE_test"].map(lambda x: f"{x:.4f}")
            df["ΔR² vs Benchmark (p.p.)"] = df["ΔR² vs Benchmark (p.p.)"].map(lambda x: f"{x:+.2f}")
            df["ΔRMSE vs Benchmark (bps)"] = df["ΔRMSE vs Benchmark (bps)"].map(lambda x: f"{x:+.1f}")
            df["DM_stat"] = df["DM_stat"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            df["DM_p"] = df["DM_p"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            print("=" * 80)
            print(f"INCREMENTAL SIGNAL REGRESSIONS — PERFORMANCE SUMMARY ({split_label} split)")
            print(f"Benchmark R²: {bench_r2:.4f}, Benchmark RMSE: {bench_rmse:.4f}")
            print("=" * 80)
            print(df.to_string(index=False))
            print()


# priced in regressions - base (t0) and temporal (t1-t10)


def summarise_priced_in_regressions(specs_dir, base_name, suffix, model_configs, all_labels, h):
    split_tags = {"": "75‑25", "_80": "80‑20", "_85": "85‑15"}; results, summary_rows = {}, []
    for model_type, l1_ratio in model_configs:
        if model_type == "elastic": model_label = f"Elastic Net (L1={l1_ratio})"; model_key = f"EN (L1={l1_ratio})"
        else: model_label = model_type.capitalize(); model_key = model_type.upper()
        split_records = {}
        for tag, slabel in split_tags.items():
            split = tag.strip("_"); split_part = f"_{split}" if split else ""
            if model_type == "elastic": base = f"{model_type}_{base_name}_t{h}{split_part}_agn_l1_{l1_ratio}_clust"
            else: base = f"{model_type}_{base_name}_t{h}{split_part}_agn_clust"
            res_path = os.path.join(specs_dir, f"{base}_results.dat"); shap_path = os.path.join(specs_dir, f"{base}_core_shap.dat"); stab_path = os.path.join(specs_dir, f"{base}_core_stability.dat")
            if not os.path.exists(res_path): continue
            res = pd.read_pickle(res_path)
            if res.empty or "R²_test" not in res: continue
            best = res.loc[res["R²_test"].idxmax()]; r2_test = best["R²_test"]; rmse_test = best["RMSE_test"]
            if r2_test < 0: r2_disp, rmse_disp = "<0", ">100"
            else: r2_disp, rmse_disp = round(r2_test, 4), round(rmse_test, 4)
            summary_rows.append({"Horizon": h, "Split": slabel, "Model": model_label, "R²_train": round(best["R²_train"], 4), "R²_test": r2_disp, "RMSE_test": rmse_disp})
            if r2_test < 0 or not os.path.exists(shap_path): continue
            shap_df = pd.read_pickle(shap_path)
            if shap_df.empty: continue
            stable = set()
            if os.path.exists(stab_path):
                stab = pd.read_pickle(stab_path)
                if not stab.empty and "core_variable" in stab: stable = set(stab["core_variable"])
            shap_df["stable"] = shap_df["core_variable"].isin(stable); shap_df["label"] = shap_df["core_variable"].apply(lambda x: all_labels.get(x, x))
            shap_df["val_str"] = shap_df.apply(lambda r: f"{r['shap_importance_pct']:.2f}{'*' if r['stable'] else ''}", axis=1)
            split_records[slabel] = shap_df[["label", "val_str", "shap_importance_pct"]].copy()
        if not split_records: continue
        merged = None
        for split, df in split_records.items():
            df = df.rename(columns={"val_str": split})
            merged = df if merged is None else pd.merge(merged, df, on="label", how="outer")
        merged.fillna(0, inplace=True)
        present_cols = [c for c in split_tags.values() if c in merged.columns]
        if not present_cols: continue
        merged["Average"] = (merged[present_cols].applymap(lambda x: float(str(x).strip("*")) if isinstance(x, (int, float, str)) and str(x).strip("*") not in ["", "-"] else 0).mean(axis=1).round(2))
        merged = merged[merged["Average"] >= 1.0]
        if merged.empty: continue
        merged = merged[["label", "Average"] + present_cols].sort_values("Average", ascending=False)
        results[model_key] = merged.set_index("label")
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values(["Horizon", "Split", "Model"])
        print("=" * 90); print("PRICED‑IN LEVEL REGRESSIONS — PERFORMANCE SUMMARY"); print("=" * 90)
        print(df_sum.to_string(index=False)); print(); print("* denotes variables selected in >80% of regressions (stability selection not performed for Ridge)"); print()
    for model_key, dfm in results.items():
        print("=" * 120); print(f"{model_key.upper()} — SHAP IMPORTANCE BY TRAIN‑TEST SPLIT (%)"); print("=" * 120)
        print(dfm.reset_index().to_string(index=False)); print()
    print("=" * 140); print("FINAL AVERAGE SHAP IMPORTANCE BY MODEL (%)"); print("=" * 140)
    all_label_set = sorted({v for df in results.values() for v in df.index}); final_df = pd.DataFrame(index=all_label_set)
    for m, dfm in results.items(): final_df[m] = dfm["Average"]
    final_df.fillna(0, inplace=True); final_df["Average"] = final_df.mean(axis=1).round(2)
    cols = ["Average"] + [c for c in ["LASSO", "RIDGE", "EN (L1=0.25)", "EN (L1=0.5)", "EN (L1=0.75)"] if c in final_df.columns]
    final_df = final_df[final_df["Average"] > 1][cols].round(2).sort_values("Average", ascending=False)
    print(final_df.to_string())


# temporal benchmark


def get_benchmark_stats(df, split_share):
    bench = []
    horizons = range(1, 11)
    for h in horizons:
        col = f"tgt_spread_t{h}"
        if col not in df:
            continue
        split = check_split_year(df, col, split_share)
        sub = df[["year", "tgt_spread", col]].dropna()
        test = sub[sub["year"] > split]
        if test.empty:
            continue
        y_true, y_pred = test[col], test["tgt_spread"]
        bench.append({
            "horizon": h,
            "R2_test": r2_score(y_true, y_pred),
            "RMSE_test": np.sqrt(mean_squared_error(y_true, y_pred))
        })
    benchmark_df = pd.DataFrame(bench)
    return benchmark_df


# temporal signal regressions


def print_temporal_signal_stats(specs_dir, base_name, models, benchmark_dfs):
    split_map = {0.75: ("", "75-25"), 0.80: ("_80", "80-20"), 0.85: ("_85", "85-15")}
    for split_share, (split_tag, split_label) in split_map.items():
        benchmark_df = benchmark_dfs.get(split_share)
        if benchmark_df is None or benchmark_df.empty: continue
        bench_r2 = benchmark_df.set_index("horizon")["R2_test"]
        bench_rmse = benchmark_df.set_index("horizon")["RMSE_test"]

        print("-" * 135)
        print(f"{split_label} TRAIN-TEST SET SPLIT".center(100))
        print("-" * 135)

        for model in models:
            variants = [0.5] if model == "elastic" else [None]
            all_rows = []
            for l1_ratio in variants:
                for h in range(0, 11):
                    if model == "elastic":
                        fname = f"{model}_{base_name}_t{h}{split_tag}_l1_{l1_ratio}_clust_results.dat"
                        model_label = f"Elastic Net (L1={l1_ratio})"
                    else:
                        fname = f"{model}_{base_name}_t{h}{split_tag}_clust_results.dat"
                        model_label = model.capitalize()
                    path = os.path.join(specs_dir, fname)
                    if not os.path.exists(path): continue
                    df = pd.read_pickle(path)
                    if df.empty or "R²_test" not in df: continue
                    best = df.loc[df["R²_test"].idxmax()]
                    bench_r2_h, bench_rmse_h = bench_r2.get(h, np.nan), bench_rmse.get(h, np.nan)
                    r2_val, rmse_val, dm_stat, dm_p = best["R²_test"], best["RMSE_test"], best.get("DM_stat"), best.get("DM_p")

                    if pd.notna(dm_p) and dm_p < 0.05:
                        if pd.notna(dm_stat) and dm_stat > 0: dm_result = "Accepted (SIGNAL)"
                        elif pd.notna(dm_stat) and dm_stat < 0: dm_result = "Accepted (NOISE)"
                        else: dm_result = "Accepted"
                    else: dm_result = "Rejected"

                    if r2_val < 0:
                        r2_disp, dr2_disp, drmse_disp = "<0", "-", "-"
                    else:
                        r2_disp = f"{r2_val:.4f}"
                        d_r2 = (r2_val - bench_r2_h) * 100 if pd.notna(bench_r2_h) else np.nan
                        d_rmse = (rmse_val - bench_rmse_h) * 100 if pd.notna(bench_rmse_h) else np.nan
                        dr2_disp = f"{d_r2:+.2f}" if pd.notna(d_r2) else "—"
                        drmse_disp = f"{d_rmse:+.1f}" if pd.notna(d_rmse) else "—"

                    rmse_disp = ">100" if rmse_val > 100 else f"{rmse_val:.4f}"
                    all_rows.append({
                        "horizon": h,
                        "R²_test": r2_disp,
                        "RMSE_test": rmse_disp,
                        "ΔR² vs Benchmark (p.p.)": dr2_disp,
                        "ΔRMSE vs Benchmark (bps)": drmse_disp,
                        "H1=Non-equal predictive accuracy (DM test)": dm_result,
                        "DM_stat": f"{dm_stat:.2f}" if pd.notna(dm_stat) else "—",
                        "DM_p": f"{dm_p:.3f}" if pd.notna(dm_p) else "—"
                    })
            if not all_rows: continue
            df = pd.DataFrame(all_rows).sort_values("horizon")
            print(f"\n>>> {model_label.upper()}")
            print("-" * 135)
            print(df.to_string(index=False))
        print("-" * 135)
        print("\n")


def plot_temporal_signal_summary(specs_dir, base_name, models, benchmark_dfs):
    split_map = {0.75: ("", "75‑25"), 0.80: ("_80", "80‑20"), 0.85: ("_85", "85‑15")}
    records = []
    for split_share, (split_tag, split_label) in split_map.items():
        for model in models:
            variants = [0.5] if model == "elastic" else [None]
            for l1_ratio in variants:
                for h in range(0, 11):
                    if model == "elastic":
                        fname = f"{model}_{base_name}_t{h}{split_tag}_l1_{l1_ratio}_clust_results.dat"
                    else:
                        fname = f"{model}_{base_name}_t{h}{split_tag}_clust_results.dat"
                    path = os.path.join(specs_dir, fname)
                    if not os.path.exists(path): continue
                    try: df = pd.read_pickle(path)
                    except Exception: continue
                    if df.empty or "R²_test" not in df: continue
                    best = df.loc[df["R²_test"].idxmax()]
                    dm_stat, dm_p = best.get("DM_stat", np.nan), best.get("DM_p", np.nan)
                    if pd.notna(dm_p) and dm_p < 0.05:
                        if pd.notna(dm_stat) and dm_stat > 0: outcome = "SIGNAL"
                        elif pd.notna(dm_stat) and dm_stat < 0: outcome = "NOISE"
                        else: outcome = "SIGNAL"
                    else: outcome = "EQUAL"
                    records.append({"horizon": h, "model": model, "outcome": outcome})
    if not records:
        print("No DM test results found for plotting.")
        return
    df = pd.DataFrame(records)
    summary = df.groupby(["horizon", "outcome"]).size().reset_index(name="count")
    colors = {"SIGNAL": "#2E86AB", "NOISE": "#E67E22", "EQUAL": "#888888"}
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    bar_width = 0.25
    x_vals = sorted(df["horizon"].unique())
    outcomes = ["SIGNAL", "NOISE", "EQUAL"]
    offsets = {"SIGNAL": -bar_width, "NOISE": 0, "EQUAL": bar_width}
    for outcome in outcomes:
        subset = summary[summary["outcome"] == outcome]
        vals = [subset.loc[subset["horizon"] == h, "count"].sum() if h in subset["horizon"].values else 0 for h in x_vals]
        xpos = np.array(x_vals) + offsets[outcome]
        ax.bar(xpos, vals, width=bar_width, color=colors[outcome], alpha=0.8, label=outcome)
    ax.set_xticks(x_vals); ax.set_xticklabels(x_vals)
    ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("Count of DM outcomes")
    ax.set_title("Forecast horizons — DM test outcome summary across all models and splits", fontsize=14)
    ax.legend(frameon=True, loc="upper center", fontsize=11)
    plt.tight_layout()
    plt.show()


def summarise_temporal_signal_vars(specs_dir, base_name, models, all_labels):
    shap_records, signal_records = [], []
    split_map = {0.75: ("", "75/25"), 0.80: ("_80", "80/20"), 0.85: ("_85", "85/15")}
    for split_share, (split_tag, split_label) in split_map.items():
        for h in range(0, 11):
            for model in models:
                l1_vals = [0.5] if model == "elastic" else [None]
                for l1 in l1_vals:
                    if model == "elastic":
                        mlabel = "Elastic Net (L1=0.5)"
                        base = f"{model}_{base_name}_t{h}{split_tag}_l1_{l1}_clust"
                    else:
                        mlabel = model.capitalize()
                        base = f"{model}_{base_name}_t{h}{split_tag}_clust"
                    resf = os.path.join(specs_dir, f"{base}_results.dat")
                    shapf = os.path.join(specs_dir, f"{base}_core_shap.dat")
                    if not os.path.exists(resf): continue
                    rdf = pd.read_pickle(resf)
                    if rdf.empty or "R²_test" not in rdf: continue
                    best = rdf.loc[rdf["R²_test"].idxmax()]
                    dm_p, dm_stat = best.get("DM_p"), best.get("DM_stat")
                    if not (pd.notna(dm_p) and dm_p < 0.05 and pd.notna(dm_stat) and dm_stat > 0): continue
                    signal_records.append({"Split": split_label, "Horizon": h, "Model": mlabel})
                    if not os.path.exists(shapf): continue
                    sdf = pd.read_pickle(shapf)
                    if sdf.empty or "shap_importance_pct" not in sdf: continue
                    for _, r in sdf.iterrows():
                        shap_records.append({"Split": split_label, "Horizon": h, "Model": mlabel, "Variable": r["core_variable"], "SHAP_pct": r["shap_importance_pct"]})
    shap_df = pd.DataFrame(shap_records)
    sig_df = pd.DataFrame(signal_records)
    shap_df["Label"] = shap_df["Variable"].apply(lambda x: all_labels.get(x, x))
    if shap_df.empty or sig_df.empty:
        print("No signal regressions found.")
        return
    full_horizon_stats = []
    for h in sorted(sig_df["Horizon"].unique()):
        sig_h = sig_df[sig_df["Horizon"] == h]
        sh_h = shap_df[shap_df["Horizon"] == h]
        if sig_h.empty: continue
        print("=" * 150)
        print(f"AVERAGE SHAP IMPORTANCE BY MODEL FOR T{h} (%)")
        print("=" * 150)
        model_splits = {"Lasso": set(), "Ridge": set(), "Elastic Net (L1=0.5)": set()}
        for _, r in sig_h.iterrows():
            for key in model_splits:
                if key.split()[0] in r["Model"]:
                    model_splits[key].add(r["Split"])
        all_vars = sorted(sh_h["Label"].unique())
        rows = []
        for var in all_vars:
            entry = {}
            for m, splits in model_splits.items():
                if not splits: entry[m] = "-"; continue
                vals = []
                for s in splits:
                    mask = (sh_h["Label"] == var) & (sh_h["Model"] == m) & (sh_h["Split"] == s)
                    val = sh_h.loc[mask, "SHAP_pct"].mean() if mask.any() else 0
                    vals.append(val)
                entry[m] = round(np.mean(vals), 2)
            numbers = [v for v in entry.values() if isinstance(v, (int, float, np.number))]
            entry["Average"] = round(np.mean(numbers), 2) if numbers else 0
            entry["Label"] = var
            rows.append(entry)
        df_all = pd.DataFrame(rows)
        for _, row in df_all.iterrows():
            full_horizon_stats.append({"Horizon": h, "Label": row["Label"], "Average": row["Average"]})
        df_show = df_all[df_all["Average"] >= 1.0].sort_values("Average", ascending=False)
        if df_show.empty:
            print(f"No variables with average SHAP >1% for T{h}.")
            continue
        fmt = lambda s: f"({', '.join(sorted(list(s)))})" if s else "(–)"
        lasso_splits, ridge_splits, en_splits = fmt(model_splits["Lasso"]), fmt(model_splits["Ridge"]), fmt(model_splits["Elastic Net (L1=0.5)"])
        rename_map = {"Lasso": f"Lasso {lasso_splits}", "Ridge": f"Ridge {ridge_splits}", "Elastic Net (L1=0.5)": f"Elastic Net (L1=0.5) {en_splits}"}
        model_cols = ["Average", f"Lasso {lasso_splits}", f"Ridge {ridge_splits}", f"Elastic Net (L1=0.5) {en_splits}"]
        df_show = df_show.rename(columns=rename_map)[["Label"] + model_cols]
        print(df_show.to_string(index=False))
    if not full_horizon_stats:
        print("No data for final horizon table.")
        return
    print("=" * 110)
    print("AVERAGE SHAP IMPORTANCE BY HORIZON (%)")
    print("=" * 110)
    avg_df = pd.DataFrame(full_horizon_stats)
    pivot = (avg_df.pivot_table(values="Average", index="Label", columns="Horizon", aggfunc="mean", fill_value=0).round(2).sort_index(axis=1))
    pivot["Average"] = pivot.mean(axis=1).round(2)
    cols = ["Average"] + [c for c in pivot.columns if c != "Average"]
    pivot = pivot[cols].sort_values("Average", ascending=False)
    pivot = pivot[pivot["Average"] > 0.5]
    print(pivot.to_string())


# temporal prediction (lagged levels)


def get_prediction_stats(specs_dir, base_name, suffix="", split_share=None):
    records = []
    horizons = range(0, 11)

    split_tag = ""
    if split_share == 0.80:
        split_tag = "_80"
    elif split_share == 0.85:
        split_tag = "_85"

    for h in horizons:
        prefix = f"{base_name}"
        if h > 0:
            prefix = f"{base_name}_t{h}"
        prefix = f"{prefix}{split_tag}"

        for model_type in ["lasso", "ridge", "elastic"]:
            if model_type == "elastic":
                l1_ratio = 0.5
                fname = f"{model_type}_{prefix}_agn_l1_{l1_ratio}_clust_results.dat"
                label = f"elastic_l1_{l1_ratio}"
            else:
                fname = f"{model_type}_{prefix}_agn_clust_results.dat"
                label = model_type

            path = os.path.join(specs_dir, fname)
            if not os.path.exists(path):
                continue

            try:
                df = pd.read_pickle(path)
                df = df.rename(columns={"R²_test": "R2_test"})
                if df.empty or "R2_test" not in df:
                    continue
                best = df.loc[df["R2_test"].idxmax()]
                records.append({
                    "horizon": h,
                    "model": label,
                    "split_share": split_share or 0.75,
                    "R2_test": best["R2_test"],
                    "RMSE_test": best["RMSE_test"],
                })
            except Exception as e:
                print(f"Error reading {path}: {e}")

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = (
            summary_df[summary_df["R2_test"].notna()]
            .sort_values(["model", "horizon"])
            .reset_index(drop=True)
        )

    return summary_df


def plot_discontinuous(ax, h, values, **kw):
    h, v = np.array(h), np.array(values)
    mask = ~np.isnan(v)
    h, v = h[mask], v[mask]
    if len(h) == 0:
        return
    gaps = np.where(np.diff(h) > 1)[0] + 1
    for xs, ys in zip(np.split(h, gaps), np.split(v, gaps)):
        ax.plot(xs, ys, **kw)


def plot_temporal_stats(summary_df, benchmark_df, split_share):
    colors = {
    "lasso": "#2E86AB",
    "ridge": "#7D3C98",
    "elastic_l1_0.5": "#E67E22"
}   
    model_labels = {
        "lasso": "Lasso",
        "ridge": "Ridge",
        "elastic_l1_0.5": "Elastic Net L1=0.5"
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    plot_df = summary_df[summary_df["R2_test"] > 0].copy()
    ax = axes[0]
    for key, label in model_labels.items():
        sub = plot_df[plot_df["model"] == key].sort_values("horizon")
        if not sub.empty:
            plot_discontinuous(ax, sub["horizon"], sub["R2_test"],
                              marker="o", markersize=4, lw=1.5,
                              color=colors[key], label=label)
    ax.plot(benchmark_df["horizon"], benchmark_df["R2_test"],
            marker="o", markersize=4, lw=1.5, ls="--", color="#555555",
            label="Mean reversion")
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_title(f"Test R² vs forecast horizon ({split_share*100:.0f}-{(1 - split_share)*100:.0f} split)", fontweight="normal")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, loc="upper right", fontsize=10)

    ax = axes[1]
    for key, label in model_labels.items():
        sub = plot_df[plot_df["model"] == key].sort_values("horizon")
        if not sub.empty:
            plot_discontinuous(ax, sub["horizon"], sub["RMSE_test"],
                              marker="o", markersize=4, lw=1.5,
                              color=colors[key], label=label)
    ax.plot(benchmark_df["horizon"], benchmark_df["RMSE_test"],
            marker="o", markersize=4, lw=1.5, ls="--", color="#555555",
            label="Mean reversion")
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel("Test RMSE", fontsize=12)
    ax.set_title(f"Test RMSE vs forecast horizon ({split_share*100:.0f}-{(1 - split_share)*100:.0f} split)", fontweight="normal")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


def print_temporal_stats(summary_df, benchmark_df=None):
    pivot_r2 = summary_df.pivot(index="horizon", columns="model", values="R2_test").round(3).copy()
    pivot_rmse = summary_df.pivot(index="horizon", columns="model", values="RMSE_test").round(3).copy()
    rename_cols = {
        "lasso": "lasso",
        "ridge": "ridge",
        "elastic_l1_0.5": "elastic net L1=0.5",
    }
    pivot_r2.rename(columns=rename_cols, inplace=True)
    pivot_rmse.rename(columns=rename_cols, inplace=True)

    if not benchmark_df.empty:
        pivot_r2["mean reversion"] = benchmark_df.set_index("horizon")["R2_test"].round(3)
        pivot_rmse["mean reversion"] = benchmark_df.set_index("horizon")["RMSE_test"].round(3)
    pivot_r2 = pivot_r2.mask(pivot_r2 < 0, "<0")
    pivot_rmse = pivot_rmse.mask(pivot_rmse > 10, ">100")

    with pd.option_context('display.float_format', '{:.3f}'.format):
        r2_lines = pivot_r2.fillna("-").to_string(index=True).split("\n")
        rmse_lines = pivot_rmse.fillna("-").to_string(index=True).split("\n")
    max_len = max(len(r2_lines), len(rmse_lines))
    r2_lines += [""] * (max_len - len(r2_lines))
    rmse_lines += [""] * (max_len - len(rmse_lines))

    print(f"{'Test‑set R² by Horizon':<60}   Test‑set RMSE by Horizon")
    print("-" * 125)
    for l1, l2 in zip(r2_lines, rmse_lines):
        print(f"{l1:<60}   {l2}")
    print("-" * 125)


def parse_filename(filepath):
    name = Path(filepath).stem
    match = re.search(r'_t(\d+)_', name)
    horizon = int(match.group(1)) if match else 0  # t0 if no t# tag
    if "_80_" in name:
        split = 0.80
    elif "_85_" in name:
        split = 0.85
    else:
        split = 0.75
    return horizon, split


def plot_model_splits(model_prefix, title_text=None, specs_dir="specs", df=None):
    specs_path = Path(specs_dir)
    records = []

    if model_prefix == "benchmark":
        for split in [0.75, 0.80, 0.85]:
            bench_df = get_benchmark_stats(df, split)
            for _, r in bench_df.iterrows():
                records.append({
                    "horizon": r["horizon"],
                    "split": split,
                    "r2": r["R2_test"],
                    "rmse": r["RMSE_test"],
                })
    else:
        if model_prefix == "elastic_levels":
            pattern = f"{model_prefix}*_agn_l1_0.5_clust_results.dat"
        else:
            pattern = f"{model_prefix}*_agn_clust_results.dat"

        for path in specs_path.glob(pattern):
            h, split = parse_filename(path)
            try:
                df_ = pd.read_pickle(path)
                if df_.empty:
                    continue
                best = df_.loc[df_["R²_test"].idxmax()]
                records.append({
                    "horizon": h,
                    "split": split,
                    "r2": best["R²_test"],
                    "rmse": best["RMSE_test"],
                })
            except Exception as e:
                print(f"Error reading {path}: {e}")

    df_plot = pd.DataFrame(records)
    df_plot = df_plot[df_plot["r2"] > 0]

    if df_plot.empty:
        print("No valid data found for plotting.")
        return

    splits = sorted(df_plot["split"].unique())
    colors = {0.75: "#2E86AB", 0.80: "#E67E22", 0.85: "#7D3C98"}
    labels = {0.75: "25%", 0.80: "20%", 0.85: "15%"}

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    title_text = title_text or model_prefix.replace("_", " ").title()

    ax = axes[0]
    for s in splits:
        part = df_plot[df_plot["split"] == s].sort_values("horizon")
        plot_discontinuous(ax, part["horizon"], part["r2"],
                           marker="o", lw=1.4, color=colors.get(s, "#555"),
                           label=labels.get(s, str(s)))
    ax.set(title=f"{title_text}\nTest R² by forecast horizon",
           xlabel="Forecast horizon (years)", ylabel="Test R²")

    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Share of test set", frameon=True)

    ax = axes[1]
    for s in splits:
        part = df_plot[df_plot["split"] == s].sort_values("horizon")
        plot_discontinuous(ax, part["horizon"], part["rmse"],
                           marker="o", lw=1.4, color=colors.get(s, "#555"),
                           label=labels.get(s, str(s)))
    ax.set(title=f"{title_text}\nTest RMSE by forecast horizon",
           xlabel="Forecast horizon (years)", ylabel="Test RMSE")

    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Share of test set", frameon=True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()