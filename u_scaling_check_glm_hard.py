import numpy as np
import argparse
import os
import sys
import math
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algorithms import betc_glm
from utils import generate_hard_glm_instances, borda_score_from_preferences, linear_link

def calculate_exploration_steps(K, U, T, d, tau_constant):
    """
    Estimates the number of exploration steps for BETC-GLM.
    This is a pre-run estimation for validation purposes.
    """
    # Phase 1: Pure random exploration
    tau = int(np.ceil(tau_constant * (d + np.log(U * T))))
    tau = min(tau, T)

    # Phase 2: Designed exploration
    eps = d**(1/6) * T**(-1/3) * U**(-1/3)
    # The number of samples is roughly d/eps^2.
    # This is a rough estimate as it depends on the G-optimal design `pi`.
    # A simpler upper bound is K*K*d/eps^2, but a factor of d is more typical.
    designed_steps = d / (eps**2)

    return int(tau + designed_steps)

def main(args):
    """
    Main function to run the U-scaling experiment for BETC-GLM on hard instances.
    """
    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)

    U_values = sorted([int(u) for u in args.U_values.split(',')])
    K = args.K
    T = args.T
    d = args.d
    run_counts = args.run_counts

    print("--- U-Scaling Experiment for BETC-GLM (Hard Instance) ---")
    print(f"Parameters: K={K}, d={d}, T={T}, Seeds={run_counts}")
    print(f"U values to sweep: {U_values}")

    # --- Validity Checks ---
    print("\n[1/5] Performing validity checks...")
    exploration_steps_map = {}
    for U in U_values:
        if K < 2**d:
            raise ValueError(f"K must be >= 2^d for hard GLM instance. Got K={K}, d={d}.")

        est_expl_steps = calculate_exploration_steps(K, U, T, d, args.tau_constant)
        exploration_steps_map[U] = est_expl_steps
        print(f"  For U={U}: Estimated exploration steps = {est_expl_steps}")
        if not est_expl_steps < T:
            raise ValueError(
                f"Error: For U={U}, the estimated exploration steps ({est_expl_steps}) "
                f"are not less than the time horizon T ({T}). "
                "Increase T or reduce exploration parameters."
            )
    print("All estimated exploration budgets are within the time horizon T. Checks passed.")

    # --- Run Simulations ---
    print("\n[2/5] Running simulations...")
    results = []
    rng_seeder = np.random.default_rng(seed=args.base_seed)
    seeds = rng_seeder.integers(0, 100000, size=run_counts)

    # For storing full regret trajectories for averaging
    all_regret_trajectories = {}
    actual_exploration_steps = {}


    for U in U_values:
        c = 640
        Delta = (11/(c*d * U**2 * T)) ** (1/3)
        while Delta*d > 1/8:
            c+=1
            Delta = (11/(c*d * U**2 * T)) ** (1/3)

        # Define the reference theta vector for the hard instance
        theta_base = np.full(d, Delta)
        print(f"  Running for U = {U}...", flush=True)
        final_regrets = np.zeros(run_counts)
        regret_curves = np.zeros((run_counts, T))
        expl_steps_for_U = np.zeros(run_counts)

        # Generate instance once per U (phi is deterministic for hard GLM)
        pref_matrices, phi_glm = generate_hard_glm_instances(U=U, K=K, d=d, Delta=Delta, theta_base=theta_base)
        s_true = borda_score_from_preferences(pref_matrices)

        for i in range(run_counts):
            s = seeds[i]
            rng = np.random.default_rng(seed=s)

            # Run algorithm
            regret, expl_steps, _, _ = betc_glm(
                P_users=pref_matrices,
                s_true=s_true,
                phi=phi_glm,
                T=T,
                rng=rng,
                iters_fw=args.iters_fw,
                tau_constant=args.tau_constant,
                goptimal_iters=args.goptimal_iters,
                link=linear_link, # Hard GLM instance uses linear link
            )
            final_regrets[i] = regret[-1]
            regret_curves[i, :] = regret
            expl_steps_for_U[i] = expl_steps

        all_regret_trajectories[U] = regret_curves
        actual_exploration_steps[U] = np.mean(expl_steps_for_U)

        mean_regret = np.mean(final_regrets)
        std_regret = np.std(final_regrets)
        ci_half_width = 1.96 * std_regret / np.sqrt(run_counts)

        results.append({
            'U': U,
            'mean_final_regret': mean_regret,
            'std_dev': std_regret,
            'ci_95_half_width': ci_half_width,
            'actual_expl_steps': actual_exploration_steps[U],
            'num_seeds': run_counts
        })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'u_scaling_glm_hard_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[3/5] Saved results to {csv_path}")

    # --- Log-Log Regression ---
    print("\n[4/5] Performing log-log regression...")
    log_U = np.log(results_df['U'])
    log_regret = np.log(results_df['mean_final_regret'])

    slope, intercept, r_value, p_value, std_err = linregress(log_U, log_regret)

    print(f"  Fit: log(Regret) = {slope:.4f} * log(U) + {intercept:.4f}")
    print(f"  Empirical Exponent (slope): {slope:.4f} ± {std_err:.4f}")
    print(f"  R-squared: {r_value**2:.4f}")

    # --- Plotting ---
    print("\n[5/5] Generating plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot empirical data with error bars
    ax.errorbar(
        results_df['U'],
        results_df['mean_final_regret'],
        yerr=results_df['ci_95_half_width'],
        fmt='o',
        capsize=5,
        label='Empirical Mean Regret (95% CI)',
        color='black',
        zorder=10
    )

    # Plot fitted line
    fit_regret = np.exp(intercept + slope * log_U)
    ax.plot(results_df['U'], fit_regret, 'r-', label=f'Fit (slope={slope:.2f})')

    # Plot reference lines (anchored to the first data point)
    anchor_U = results_df['U'].iloc[0]
    anchor_regret = results_df['mean_final_regret'].iloc[0]

    ref_2_3 = anchor_regret * (results_df['U'] / anchor_U)**(2/3)
    ax.plot(results_df['U'], ref_2_3, 'b--', label='Reference (slope=2/3)')

    ref_1_3 = anchor_regret * (results_df['U'] / anchor_U)**(1/3)
    ax.plot(results_df['U'], ref_1_3, 'g--', label='Reference (slope=1/3)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Users (U)')
    ax.set_ylabel('Final Cumulative Regret at T')
    ax.set_title(f'Regret Scaling with U for Hard GLM Instance (K={K}, d={d}, T={T})')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plot_path = os.path.join(args.output_dir, 'u_scaling_glm_hard_loglog_plot.pdf')
    fig.savefig(plot_path, bbox_inches='tight')
    print(f"  Saved log-log plot to {plot_path}")
    plt.close(fig)

    # Plot average regret trajectories with confidence intervals
    fig_traj, ax_traj = plt.subplots(figsize=(10, 7))
    for U, trajectories in all_regret_trajectories.items():
        mean_traj = np.mean(trajectories, axis=0)
        std_traj = np.std(trajectories, axis=0)
        ci_half_width = 1.96 * std_traj / np.sqrt(run_counts)

        line, = ax_traj.plot(mean_traj, label=f'U={U}')
        ax_traj.fill_between(np.arange(T), mean_traj - ci_half_width, mean_traj + ci_half_width, color=line.get_color(), alpha=0.2)

    ax_traj.set_xlabel('Rounds (t)')
    ax_traj.set_ylabel('Cumulative Regret')
    ax_traj.set_title(f'Average Regret Trajectories ({run_counts} seeds per U)')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.5)
    traj_plot_path = os.path.join(args.output_dir, 'u_scaling_glm_hard_trajectories.pdf')
    fig_traj.savefig(traj_plot_path, bbox_inches='tight')
    print(f"  Saved trajectory plot to {traj_plot_path}")
    plt.close(fig_traj)

    # --- Final Summary ---
    print("\n--- Summary ---")
    print(f"Fitted exponent: {slope:.4f} ± {std_err:.4f}")
    dist_to_2_3 = abs(slope - 2/3)
    dist_to_1_3 = abs(slope - 1/3)
    print(f"Tabular theoretical upper bound exponent: {2/3:.4f}")
    print(f"Tabular theoretical lower bound exponent: {1/3:.4f}")
    if dist_to_2_3 < dist_to_1_3:
        print("The empirical exponent is closer to 2/3.")
    else:
        print("The empirical exponent is closer to 1/3.")
    print("-------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-scaling experiment for BETC-GLM on hard instances.")

    parser.add_argument("--U_values", type=str, default="4,8,16,32,64,128", help="Comma-separated list of U values to test.")
    parser.add_argument("--K", type=int, default=16, help="Number of arms. Must be >= 2^d.")
    parser.add_argument("--d", type=int, default=3, help="Dimension for hard GLM instance.")
    parser.add_argument("--T", type=int, default=600000, help="Time horizon.")
    parser.add_argument("--run_counts", type=int, default=20, help="Number of independent seeds per U value.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for generating run seeds.")

    # Algorithm Hyperparameters
    parser.add_argument("--tau_constant", type=float, default=100.0, help="Constant for pure exploration phase.")
    parser.add_argument("--goptimal_iters", type=int, default=50, help="Iterations for G-optimal design.")
    parser.add_argument("--iters_fw", type=int, default=300, help="Frank-Wolfe iterations for the algorithm.")

    parser.add_argument("--output_dir", type=str, default="experiments/results/u_scaling_glm_hard/", help="Directory to save results.")

    args = parser.parse_args()
    main(args)