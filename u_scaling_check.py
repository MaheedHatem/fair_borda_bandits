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

from algorithms import explore_then_commit_borda
from utils import generate_hard_instances, borda_score_from_preferences

def calculate_L_expl(K, U, T, exploration_frac):
    """
    Calculates the number of exploration duels per arm (L).
    This logic is extracted from explore_then_commit_borda for pre-run validation.
    """
    if K <= 0 or U <= 0 or T <= 0:
        return 0
    log_term = math.log(U * K * T) if U * K * T > 1 else 0
    L = int(exploration_frac * (K**(-2/3) * T**(2/3) * U**(2/3) * log_term**(1/3)))
    return L

def main(args):
    """
    Main function to run the U-scaling experiment for Fair-BETC.
    """
    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    U_values = sorted([int(u) for u in args.U_values.split(',')])
    K = args.K
    T = args.T
    run_counts = args.run_counts
    exploration_frac = args.exploration_frac

    print("--- U-Scaling Experiment for Fair-BETC (Tabular) ---")
    print(f"Parameters: K={K}, T={T}, Seeds={run_counts}, exploration_frac={exploration_frac}")
    print(f"U values to sweep: {U_values}")

    # --- Validity Checks ---
    print("\n[1/5] Performing validity checks...")
    L_values = {}
    for U in U_values:
        L = calculate_L_expl(K, U, T, exploration_frac)
        L_values[U] = L
        total_exploration_steps = K * L
        print(f"  For U={U}: L (duels per arm) = {L}, Total exploration steps = {total_exploration_steps}")
        if not total_exploration_steps < T:
            raise ValueError(
                f"Error: For U={U}, the total exploration steps ({total_exploration_steps}) "
                f"are not less than the time horizon T ({T}). "
                "Increase T or reduce the exploration fraction."
            )
    print("All exploration budgets are within the time horizon T. Checks passed.")

    # --- Run Simulations ---
    print("\n[2/5] Running simulations...")
    results = []
    rng_seeder = np.random.default_rng(seed=args.base_seed)
    seeds = rng_seeder.integers(0, 100000, size=run_counts)

    # For storing full regret trajectories for averaging
    all_regret_trajectories = {}

    for U in U_values:
        print(f"  Running for U = {U}...")
        final_regrets = np.zeros(run_counts)
        regret_curves = np.zeros((run_counts, T))
        
        for i in range(run_counts):
            s = seeds[i]
            rng = np.random.default_rng(seed=s)
            
            # Generate hard instance (reference case, m_target=None)
            
            c = 64
            epsilon = (K/(c*U**2*T))**(1/3)
            while epsilon > 0.05:
                c+=1
                epsilon = (K/(c*U**2*T))**(1/3)

            pref_matrices = generate_hard_instances(U=U, K=K, epsilon=epsilon, m_target=None)
            s_true = borda_score_from_preferences(pref_matrices)

            # Run algorithm
            regret, _, _, _ = explore_then_commit_borda(
                P_users=pref_matrices,
                s_true=s_true,
                T=T,
                rng=rng,
                iters_fw=args.iters_fw,
                exploration_frac=exploration_frac
            )
            
            final_regrets[i] = regret[-1]
            regret_curves[i, :] = regret

        all_regret_trajectories[U] = regret_curves

        mean_regret = np.mean(final_regrets)
        std_regret = np.std(final_regrets)
        ci_half_width = 1.96 * std_regret / np.sqrt(run_counts)
        
        results.append({
            'U': U,
            'mean_final_regret': mean_regret,
            'std_dev': std_regret,
            'ci_95_half_width': ci_half_width,
            'L_expl': L_values[U],
            'num_seeds': run_counts
        })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'u_scaling_results.csv')
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

    # Plot reference lines
    # Anchor point: first data point
    anchor_U = results_df['U'].iloc[0]
    anchor_regret = results_df['mean_final_regret'].iloc[0]

    # Reference line for slope = 2/3
    ref_2_3 = anchor_regret * (results_df['U'] / anchor_U)**(2/3)
    ax.plot(results_df['U'], ref_2_3, 'b--', label='Reference (slope=2/3)')

    # Reference line for slope = 1/3
    ref_1_3 = anchor_regret * (results_df['U'] / anchor_U)**(1/3)
    ax.plot(results_df['U'], ref_1_3, 'g--', label='Reference (slope=1/3)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Users (U)')
    ax.set_ylabel('Final Cumulative Regret at T')
    ax.set_title(f'Regret Scaling with U (K={K}, T={T})')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plot_path = os.path.join(args.output_dir, 'u_scaling_loglog_plot.pdf')
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
    traj_plot_path = os.path.join(args.output_dir, 'u_scaling_trajectories.pdf')
    fig_traj.savefig(traj_plot_path, bbox_inches='tight')
    print(f"  Saved trajectory plot to {traj_plot_path}")
    plt.close(fig_traj)

    # --- Final Summary ---
    print("\n--- Summary ---")
    print(f"Fitted exponent: {slope:.4f} ± {std_err:.4f}")
    dist_to_2_3 = abs(slope - 2/3)
    dist_to_1_3 = abs(slope - 1/3)
    print(f"Theoretical upper bound exponent: {2/3:.4f}")
    print(f"Theoretical lower bound exponent: {1/3:.4f}")
    if dist_to_2_3 < dist_to_1_3:
        print("The empirical exponent is closer to 2/3.")
    else:
        print("The empirical exponent is closer to 1/3.")
    print("-------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-scaling experiment for Fair-BETC.")
    
    parser.add_argument("--U_values", type=str, default="4,8,16,32,64,128", help="Comma-separated list of U values to test.")
    parser.add_argument("--K", type=int, default=11, help="Number of arms.")
    parser.add_argument("--T", type=int, default=200000, help="Time horizon.")
    parser.add_argument("--run_counts", type=int, default=20, help="Number of independent seeds per U value.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for generating run seeds.")
    
    parser.add_argument("--exploration_frac", type=float, default=0.25, help="Constant factor for exploration length L.")
    parser.add_argument("--iters_fw", type=int, default=300, help="Frank-Wolfe iterations for the algorithm.")
    
    parser.add_argument("--output_dir", type=str, default="experiments/results/u_scaling/", help="Directory to save results.")

    args = parser.parse_args()
    main(args)