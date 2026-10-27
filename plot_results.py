import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def compute_user_utilities(scores_1: np.ndarray, scores_2: np.ndarray) -> np.ndarray:
    total_scores = 0.5*scores_1 + 0.5*scores_2
    return np.cumsum(total_scores, axis=1)

def min_welfare(user_utilities: np.ndarray) -> np.ndarray:
    return np.min(user_utilities, axis=2)

def utlitarian_welfare(user_utilities: np.ndarray) -> np.ndarray:
    return np.mean(user_utilities, axis=2)

def nsw_welfare(user_utilities: np.ndarray) -> np.ndarray:
    U = user_utilities.shape[2]
    return np.prod(user_utilities, axis=2) ** (1.0 / U)

def gini_welfare(user_utilities: np.ndarray) -> np.ndarray:
    runs, T, U = user_utilities.shape
    sum_abs_diff = np.zeros((runs, T))
    for i in range(U):
        for j in range(U):
            sum_abs_diff += np.abs(user_utilities[:, :, i] - user_utilities[:, :, j])
    sum_xi = np.sum(user_utilities, axis=2)
    gini = sum_abs_diff / (2 * U * sum_xi)
    return np.nan_to_num(gini, nan=0.0)

def plot_with_ci(ax, data, label, color, linestyle="-"):
    run_counts, T = data.shape
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96 * std / np.sqrt(run_counts)
    ax.plot(mean, label=label, color=color, linestyle=linestyle, rasterized=True)
    ax.fill_between(np.arange(T), mean - ci, mean + ci, color=color, alpha=0.2, rasterized=True)

def plot_results_from_folder(folder, output_dir=None, suffix=None):
    if output_dir is None:
        output_dir = folder

    algorithms = {}
    # Load Borda-ETC data
    try:
        algorithms['Fair-BETC'] = {
            'regret': np.load(os.path.join(folder, "etc_regret.npy")),
            'scores_1': np.load(os.path.join(folder, "etc_scores_1.npy")),
            'scores_2': np.load(os.path.join(folder, "etc_scores_2.npy")),
            'color': "#0072B2"  # Blue
        }
        run_counts, T, U = algorithms['Fair-BETC']['scores_1'].shape
    except FileNotFoundError:
        print(f"Borda-ETC data not found in {folder}, skipping.")
        # If the main algorithm is not there, no point in continuing
    try:
        algorithms['Fair-BETC-GLM'] = {
        'regret': np.load(os.path.join(folder, "glm_regret.npy")),
        'scores_1': np.load(os.path.join(folder, "glm_scores_1.npy")),
        'scores_2': np.load(os.path.join(folder, "glm_scores_2.npy")),
        'color': "#0072B2"  # Blue
        }
    except FileNotFoundError:
        print(f"BETC-GLM data not found in {folder}, skipping.")
    try:
        algorithms['Winner-Commit-GLM'] = {
        'regret': np.load(os.path.join(folder, "glm_winner_regret.npy")),
        'scores_1': np.load(os.path.join(folder, "glm_winner_scores_1.npy")),
        'scores_2': np.load(os.path.join(folder, "glm_winner_scores_2.npy")),
        'color': "#D55E00"  # Orange
        }
    except FileNotFoundError:
        print(f"BETC-GLM-Winner data not found in {folder}, skipping.")

        # If the main algorithm is not there, no point in continuing
    # Load Borda-Winner-ETC data if available
    try:
        algorithms['Winner-Commit'] = {
            'regret': np.load(os.path.join(folder, "etc_winner_regret.npy")),
            'scores_1': np.load(os.path.join(folder, "etc_winner_scores_1.npy")),
            'scores_2': np.load(os.path.join(folder, "etc_winner_scores_2.npy")),
            'color': "#D55E00"  # Orange
        }
    except FileNotFoundError:
        print(f"Borda-Winner-ETC data not found in {folder}, skipping.")
    
    try:
        algorithms['Utilitarian'] = {
            'regret': np.load(os.path.join(folder, "etc_util_regret.npy")),
            'scores_1': np.load(os.path.join(folder, "etc_util_scores_1.npy")),
            'scores_2': np.load(os.path.join(folder, "etc_util_scores_2.npy")),
            'color': "#83CC79"  # Green
        }
    except FileNotFoundError:
        print(f"Borda-ETC Utilitarian data not found in {folder}, skipping.")

    try:
        algorithms['Uniform-Winners'] = {
            'regret': np.load(os.path.join(folder, "etc_uniform_winners_regret.npy")),
            'scores_1': np.load(os.path.join(folder, "etc_uniform_winners_scores_1.npy")),
            'scores_2': np.load(os.path.join(folder, "etc_uniform_winners_scores_2.npy")),
            'color': "#000000"  # Black
        }
    except FileNotFoundError:
        print(f"Borda-ETC Uniform Winners data not found in {folder}, skipping.")

    try:
        algorithms['Utilitarian-GLM'] = {
            'regret': np.load(os.path.join(folder, "glm_util_regret.npy")),
            'scores_1': np.load(os.path.join(folder, "glm_util_scores_1.npy")),
            'scores_2': np.load(os.path.join(folder, "glm_util_scores_2.npy")),
            'color': "#83CC79"  # Green
        }
    except FileNotFoundError:
        print(f"BETC-GLM Utilitarian data not found in {folder}, skipping.")

    try:
        algorithms['Uniform-Winners-GLM'] = {
            'regret': np.load(os.path.join(folder, "glm_uniform_winners_regret.npy")),
            'scores_1': np.load(os.path.join(folder, "glm_uniform_winners_scores_1.npy")),
            'scores_2': np.load(os.path.join(folder, "glm_uniform_winners_scores_2.npy")),
            'color': "#000000"  # Black
        }
    except FileNotFoundError:
        print(f"BETC-GLM Uniform Winners data not found in {folder}, skipping.")
    
    if len(algorithms) == 0:
        print(f"No primary algorithm data (GLM or Borda-ETC) found in {folder}. Skipping folder.")
        return
    # Update matplotlib parameters to use LaTeX and Times font
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{times} \usepackage{amsmath}",
        "axes.labelsize": 24,
        "font.size": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    # Compute utilities for all loaded algorithms
    for name, data in algorithms.items():
        data['utilities'] = compute_user_utilities(data['scores_1'], data['scores_2'])

    # Plot Regret
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in algorithms.items():
        plot_with_ci(ax, data['regret'], name, data['color'])
    
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    reg_filename = f"regret_comparison_{suffix}" if suffix else "regret_comparison"
    reg_path = os.path.join(output_dir, reg_filename)
    
    #for ext in [".pdf", ".svg", ".png"]:
    for ext in [".pdf"]:
        fig.savefig(f"{reg_path}{ext}")
    plt.close(fig)
    print(f"Saved plot to {reg_path}")

    # Plot Welfare Metrics
    welfare_metrics = {
        "Minimum User Welfare": min_welfare,
        "NSW User Welfare": nsw_welfare,
        "Utilitarian User Welfare": utlitarian_welfare,
        "Gini Coefficient": gini_welfare,
    }
    filenames = {
        "Minimum User Welfare": "min_welfare_comparison",
        "NSW User Welfare": "nsw_welfare_comparison",
        "Utilitarian User Welfare": "utilitarian_welfare_comparison",
        "Gini Coefficient": "gini_coefficient",
    }

    for metric_name, metric_func in welfare_metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, data in algorithms.items():
            metric_values = metric_func(data['utilities'])
            plot_with_ci(ax, metric_values, name, data['color'])
            # store for table
            data[metric_name] = metric_values[:, -1]
        ax.set_xlabel("Rounds")
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        filename = f"{filenames[metric_name]}_{suffix}" if suffix else f"{filenames[metric_name]}"
        path = os.path.join(output_dir, filename)
        #for ext in [".pdf", ".svg", ".png"]:
        for ext in [".pdf"]:
            fig.savefig(f"{path}{ext}")
        plt.close(fig)
        print(f"Saved plot to {path}")

    # --- Save tables ---
    metrics_summary = {}
    all_metrics = ["Cumulative Regret"] + list(welfare_metrics.keys())
    for metric_name in all_metrics:
        metrics_summary[metric_name] = {}
        for alg_name, alg_data in algorithms.items():
            if metric_name == "Cumulative Regret":
                metrics_summary[metric_name][alg_name] = alg_data['regret'][:, -1]
            else:
                metrics_summary[metric_name][alg_name] = alg_data[metric_name]

    table_filename = f"metrics_table_{suffix}.txt" if suffix else "metrics_table.txt"
    table_path = os.path.join(output_dir, table_filename)
    with open(table_path, "w") as f:
        for metric_name, alg_data in metrics_summary.items():
            f.write(f"Metric: {metric_name}\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'Algorithm':<25} | {'Mean':<12} | {'CI':<12}\n")
            f.write("-" * 55 + "\n")
            for alg_name, data in alg_data.items():
                mean_val = np.mean(data)
                std_val = np.std(data)
                n_runs = len(data)
                ci_val = 1.96 * std_val / np.sqrt(n_runs)
                f.write(f"{alg_name:<25} | {mean_val:<12.4f} | {ci_val:<12.4f}\n")
            f.write("\n")
    print(f"Saved metrics table to: {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from folder")
    parser.add_argument("folder", type=str, help="Folder containing .npy files")
    parser.add_argument("--output_dir", type=str, help="Output directory for figures", default=None)
    parser.add_argument("--suffix", type=str, help="Suffix for filenames", default=None)
    args = parser.parse_args()
    
    plot_results_from_folder(args.folder, args.output_dir, args.suffix)
    
    plt.show()