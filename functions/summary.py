import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from functions.utils import check_split_year


def get_label(var_name, all_labels):
    return all_labels.get(var_name, var_name)


def simplify_label(label):
    base = re.sub(r'\(.*?\)', '', label)
    base = base.split('.')[-1].strip()
    return base


def summarise_signal_regressions(specs_dir, base_name, models):
    summary_data = []
    for model in models:
        for l1_ratio in ([0.25, 0.5, 0.75] if model == "elastic" else [None]):
            label = f"{model.capitalize()}" if l1_ratio is None else f"Elastic Net (L1={l1_ratio})"
            file_name = (f"{model}_{base_name}_l1_{l1_ratio}_clust_results.dat"
                          if model == "elastic"
                          else f"{model}_{base_name}_clust_results.dat")
            path = os.path.join(specs_dir, file_name)
            if not os.path.exists(path):
                continue
            df = pd.read_pickle(path)
            best = df.loc[df["R²_test"].idxmax()]
            reject = "Accepted" if (pd.notna(best["DM_p"]) and best["DM_p"] < 0.05) else "Rejected"
            summary_data.append({
                "Model": label,
                "R²_test": f"{best['R²_test']:.4f}",
                "RMSE_test": f"{best['RMSE_test']:.4f}",
                "ΔR² vs Benchmark (p.p.)": f"{(best['R²_test'] - 0.892) * 100:+.2f}",
                "ΔRMSE vs Benchmark (bps)": f"{(best['RMSE_test'] - 1.32) * 100:+.1f}",
                "H1=Incremental signal present": reject,
                "DM_stat": f"{best['DM_stat']:.2f}" if pd.notna(best['DM_stat']) else "—",
                "DM_p": f"{best['DM_p']:.3f}" if pd.notna(best['DM_p']) else "—",
            })
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df["R²_test"] = summary_df["R²_test"].astype(float)
        summary_df = summary_df.sort_values("Model", ascending=False)
        print("=" * 80)
        print("INCREMENTAL SIGNAL REGRESSIONS — PERFORMANCE SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("\nNote: ΔR² values are in percentage points (p.p.), ΔRMSE values in basis points (bps) relative to the benchmark.")


def summarise_priced_in_regressions(specs_dir, base_name, suffix, model_configs, all_labels):
    summary_rows = []
    all_top_vars = []
    for model_type, l1_ratio in model_configs:
        if model_type == 'elastic':
            model_label = f"Elastic Net (L1={l1_ratio})"
            file_base = f"{model_type}_{base_name}{suffix}_l1_{l1_ratio}_clust"
        else:
            model_label = model_type.capitalize()
            file_base = f"{model_type}_{base_name}{suffix}_clust"
        
        results_path = os.path.join(specs_dir, f"{file_base}_results.dat")
        core_shap_path = os.path.join(specs_dir, f"{file_base}_core_shap.dat")
        core_stability_path = os.path.join(specs_dir, f"{file_base}_core_stability.dat")
        results_df = pd.read_pickle(results_path)
        best_idx = results_df['R²_test'].idxmax()
        best = results_df.iloc[best_idx]
        core_shap = pd.read_pickle(core_shap_path)
        stable_cores = set()
        
        if os.path.exists(core_stability_path):
            core_stability = pd.read_pickle(core_stability_path)
            stable_cores = set(core_stability['core_variable'].values)

        top5 = core_shap.head(5).copy()
        top5['stable'] = top5['core_variable'].isin(stable_cores)
        top5['model'] = model_label

        summary_rows.append({'Model': model_label,
                             'R²_test': best['R²_test'],
                             'R²_train': best['R²_train'],
                             'RMSE_test': best['RMSE_test']})

        for _, row in top5.iterrows():
            var_name = row['core_variable'] + ('*' if row['stable'] else '')
            all_top_vars.append({'Model': model_label,
                                 'Variable': var_name,
                                 'SHAP': row['shap_importance'],
                                 'SHAP_pct': row['shap_importance_pct']})

    summary_df = pd.DataFrame(summary_rows)
    
    print("="*80)
    print("PRICED-IN LEVELS - REGRESSIONS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("\n" + "="*80)
    print("TOP 5 VARIABLES BY SHAP IMPORTANCE (per model)")
    print("="*80)
    print("* Indicates variable selected in more than 80% of stability selection regressions\n(not performed for Ridge)")

    top_vars_df = pd.DataFrame(all_top_vars)
    for model in top_vars_df['Model'].unique():
        print(f"\n{model}:")
        model_vars = top_vars_df[top_vars_df['Model'] == model]
        
        for _, row in model_vars.iterrows():
            var_core = row['Variable'].replace('*', '')
            var_label = simplify_label(get_label(var_core, all_labels))
            stability_marker = '*' if '*' in row['Variable'] else ''
            print(f"  {var_label}{stability_marker}: {row['SHAP']:.4f} ({row['SHAP_pct']:.2f}%)")
    
    shap_pivot = pd.DataFrame(all_top_vars)
    shap_pivot['core_variable'] = shap_pivot['Variable'].str.replace('*', '', regex=False)
    all_core_vars = shap_pivot['core_variable'].unique()
    shap_detail = pd.DataFrame(all_top_vars)
    shap_detail['core_variable'] = shap_detail['Variable'].str.replace('*', '', regex=False)
    shap_detail['label'] = shap_detail['core_variable'].apply(lambda x: simplify_label(get_label(x, all_labels)))

    complete_shap_data = []
    for model_type, l1_ratio in model_configs:
        if model_type == 'elastic':
            model_label = f"EN (L1={l1_ratio})"
            file_base = f"{model_type}_{base_name}{suffix}_l1_{l1_ratio}_clust"
        else:
            model_label = model_type.upper()
            file_base = f"{model_type}_{base_name}{suffix}_clust"
        
        core_shap_path = os.path.join(specs_dir, f"{file_base}_core_shap.dat")
        full_core_shap = pd.read_pickle(core_shap_path)
        relevant_vars = full_core_shap[full_core_shap['core_variable'].isin(all_core_vars)].copy()
        relevant_vars['Model'] = model_label
        relevant_vars['label'] = relevant_vars['core_variable'].apply(lambda x: simplify_label(get_label(x, all_labels)))
        complete_shap_data.append(relevant_vars[['label', 'Model', 'shap_importance_pct']])

    complete_df = pd.concat(complete_shap_data, ignore_index=True)
    pivot_table = complete_df.pivot_table(values='shap_importance_pct',
                                          index='label',
                                          columns='Model',
                                          fill_value=0.0).round(2)
    pivot_table['Average'] = pivot_table.mean(axis=1).round(2)
    pivot_table = pivot_table.sort_values('Average', ascending=False)
    desired_order = ['RIDGE', 'EN (L1=0.25)', 'EN (L1=0.5)', 'EN (L1=0.75)', 'LASSO']
    existing_cols = [c for c in desired_order if c in pivot_table.columns]
    remaining_cols = [c for c in pivot_table.columns if c not in existing_cols]
    pivot_table = pivot_table[existing_cols + remaining_cols]

    print("\n" + "="*80)
    print("SHAP IMPORTANCE BY MODEL AND VARIABLE (%)")
    print("="*80)
    print(pivot_table.to_string())


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


def get_prediction_stats(specs_dir, base_name, suffix=""):
    records = []
    horizons = range(1, 11)
    for h in horizons:
        prefix = f"{base_name}_t{h}"
        for model_type in ["lasso", "ridge", "elastic"]:
            if model_type == "elastic":
                l1_ratio = 0.5
                fname = f"{model_type}_{prefix}_agn_l1_{l1_ratio}_clust_results.dat"
                label = f"elastic_l1_{l1_ratio}"
            else:
                fname = f"{model_type}_{prefix}{suffix}_results.dat"
                label = model_type

            path = os.path.join(specs_dir, fname)
            if not os.path.exists(path):
                continue

            df = pd.read_pickle(path)
            df = df.rename(columns={"R²_test": "R2_test"})  # ensure ASCII
            best = df.loc[df["R2_test"].idxmax()]
            records.append({
                "horizon": h,
                "model": label,
                "R2_test": best["R2_test"],
                "RMSE_test": best["RMSE_test"]
            })

    summary_df = pd.DataFrame(records)
    summary_df = summary_df[summary_df["R2_test"].notna()].sort_values(["model", "horizon"])
    
    return summary_df


def plot_temporal_stats(summary_df, benchmark_df):
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
    def plot_discontinuous(ax, h, values, **kw):
        h, v = np.array(h), np.array(values)
        mask = ~np.isnan(v)
        h, v = h[mask], v[mask]
        if len(h) == 0:
            return
        gaps = np.where(np.diff(h) > 1)[0] + 1
        for xs, ys in zip(np.split(h, gaps), np.split(v, gaps)):
            ax.plot(xs, ys, **kw)

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
    ax.set_title("Test R² vs forecast horizon", fontweight="normal")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0,)
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
    ax.set_title("Test RMSE vs forecast horizon", fontweight="normal")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0,)
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

