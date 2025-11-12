import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from functions.utils import check_split_year


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
            

def summarise_priced_in_regressions(specs_dir, base_name, suffix,
                                    model_configs, all_labels, h):
    split_tags = {"": "75‑25", "_80": "80‑20", "_85": "85‑15"}
    results, summary_rows = {}, []
    for model_type, l1_ratio in model_configs:
        if model_type == "elastic":
            model_label = f"Elastic Net (L1 = {l1_ratio})"
            model_key = f"EN (L1 = {l1_ratio})"
        else:
            model_label = model_type.capitalize()
            model_key = model_type.upper()
        split_records = {}
        for tag, slabel in split_tags.items():
            split = tag.strip("_")
            split_part = f"_{split}" if split else ""
            if model_type == "elastic":
                base = f"{model_type}_{base_name}_t{h}{split_part}_agn_l1_{l1_ratio}_clust"
            else:
                base = f"{model_type}_{base_name}_t{h}{split_part}_agn_clust"
            res_path = os.path.join(specs_dir, f"{base}_results.dat")
            shap_path = os.path.join(specs_dir, f"{base}_core_shap.dat")
            stab_path = os.path.join(specs_dir, f"{base}_core_stability.dat")
            if not (os.path.exists(res_path) and os.path.exists(shap_path)):
                continue
            res = pd.read_pickle(res_path)
            if res.empty or "R²_test" not in res: continue
            best = res.loc[res["R²_test"].idxmax()]
            summary_rows.append({
                "Horizon": h,
                "Split": slabel,
                "Model": model_label,
                "R²_train": round(best["R²_train"], 4),
                "R²_test": round(best["R²_test"], 4),
                "RMSE_test": round(best["RMSE_test"], 4)
            })
            shap_df = pd.read_pickle(shap_path)
            if shap_df.empty: continue
            stable = set()
            if os.path.exists(stab_path):
                stab = pd.read_pickle(stab_path)
                if not stab.empty and "core_variable" in stab:
                    stable = set(stab["core_variable"])
            shap_df["stable"] = shap_df["core_variable"].isin(stable)
            shap_df["label"] = shap_df["core_variable"].apply(lambda x: all_labels.get(x, x))
            shap_df["val_str"] = shap_df.apply(
                lambda r: f"{r['shap_importance_pct']:.2f}{'*' if r['stable'] else ''}", axis=1
            )
            split_records[slabel] = shap_df[["label", "val_str", "shap_importance_pct"]].copy()
        if not split_records:
            continue
        merged = None
        for split, df in split_records.items():
            df = df.rename(columns={"val_str": split})
            merged = df if merged is None else pd.merge(merged, df, on="label", how="outer")
        merged.fillna(0, inplace=True)
        present_cols = [c for c in split_tags.values() if c in merged.columns]
        if not present_cols:
            continue
        merged["Average"] = (
            merged[present_cols]
            .applymap(
                lambda x: float(str(x).strip("*"))
                if isinstance(x, (int, float, str))
                and str(x).strip("*") not in ["", "-"]
                else 0
            )
            .mean(axis=1)
            .round(2)
        )
        merged = merged[merged["Average"] >= 1.0]
        if merged.empty:
            continue
        merged = merged[["label", "Average"] + present_cols].sort_values("Average", ascending=False)
        results[model_key] = merged.set_index("label")
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values(["Horizon", "Split", "Model"])
        print("=" * 90)
        print("PRICED‑IN LEVEL REGRESSIONS — PERFORMANCE SUMMARY")
        print("=" * 90)
        print(df_sum.to_string(index=False))
        print()
        print("* denotes variables selected in more than 80% of regressions from stability selection section (not performed for Ridge)")
        print()
    for model_key, dfm in results.items():
        print("=" * 120)
        print(f"{model_key.upper()} — SHAP IMPORTANCE BY TRAIN‑TEST SPLIT (%)")
        print("=" * 120)
        print(dfm.reset_index().to_string(index=False))
        print()
    print("=" * 140)
    print("FINAL AVERAGE SHAP IMPORTANCE BY MODEL (%)")
    print("=" * 140)
    all_label_set = sorted({v for df in results.values() for v in df.index})
    final_df = pd.DataFrame(index=all_label_set)
    for m, dfm in results.items():
        final_df[m] = dfm["Average"]
    final_df.fillna(0, inplace=True)
    final_df["Average"] = final_df.mean(axis=1).round(2)
    cols = ["Average"] + [
        c for c in ["LASSO", "RIDGE", "EN (L1 = 0.25)", "EN (L1 = 0.5)", "EN (L1 = 0.75)"]
        if c in final_df.columns
    ]
    final_df = final_df[final_df['Average']>1][cols].round(2).sort_values("Average", ascending=False)
    print(final_df.to_string())


# temporal


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