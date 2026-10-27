import numpy as np
from numpy.random import Generator
import argparse
from algorithms import explore_then_commit_borda, explore_then_commit_borda_winner, betc_glm, betc_glm_winner, betc_glm_uniform_winners, explore_then_commit_borda_uniform_winners
from utils import generate_user_prefs, load_sushi_data, borda_score_from_preferences, generate_random_glm_instance, logistic, linear_link
import os
import datetime
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dueling Bandits Simulation with Borda Score Objective")
    
    # Simulation Parameters
    parser.add_argument("--U", type=int, default=8, help="Number of users")
    parser.add_argument("--K", type=int, default=10, help="Number of arms")
    parser.add_argument("--T", type=int, default=10000, help="Total time horizon")
    parser.add_argument("--run_counts", type=int, default=10, help="Number of independent seeds/trials to run")
    
    # Instance Type
    parser.add_argument("--use_sushi_data", action="store_true", help="If set, use the Sushi dataset. U is used as n_clusters.")
    parser.add_argument("--sushi_as_glm", action="store_true", help="If using sushi data, treat it as a GLM instance (requires feature extraction).")
    parser.add_argument("--use_conflicting_glm_instance", action="store_true", help="If set, use GLM instance with conflicting user groups.")
    parser.add_argument("--use_glm_instance", action="store_true", help="If set, use GLM instance generation instead of random user prefs.")
    parser.add_argument("--d", type=int, default=5, help="Dimension of context vectors for GLM instance (if --use_glm_instance).")
    parser.add_argument("--theta_scale", type=float, default=1.0, help="Scale for theta_star in GLM instance (if --use_glm_instance).")
    parser.add_argument("--glm_noise_scale", type=float, default=0.1, help="Noise scale for conflicting GLM instance.")
    parser.add_argument("--glm_groups", type=int, default=2, help="Number of user groups for conflicting GLM instance.")
    parser.add_argument("--glm_link", type=str, choices=['logistic', 'linear'], default='logistic', help="Link function for GLM instance and algorithm.")

    # Algorithm Hyperparameters
    parser.add_argument("--exploration_frac", type=float, default=0.25, help="Fraction of T to use for exploration in Borda-ETC")
    parser.add_argument("--iters_fw", type=int, default=300, help="Frank-Wolfe iterations")
    
    # BETC-GLM specific hyperparameters
    parser.add_argument("--tau_constant", type=float, default=100.0, help="Tau constant for betc_glm exploration phase.")
    parser.add_argument("--goptimal_iters", type=int, default=500, help="Iterations for G-optimal design in betc_glm.")
    parser.add_argument("--base_directory", type=str, default="./results/", help="Base directory for saving results")

    args = parser.parse_args()

    # Assign to variables
    U = args.U
    K = args.K
    T = args.T
    base_directory = args.base_directory
    run_counts = args.run_counts
    
    # Algorithm Hyperparameters
    exploration_frac = args.exploration_frac
    iters_fw = args.iters_fw

    # Instance specific parameters
    use_sushi_data = args.use_sushi_data
    sushi_as_glm = args.sushi_as_glm
    use_conflicting_glm_instance = args.use_conflicting_glm_instance
    use_glm_instance = args.use_glm_instance
    d = args.d
    glm_noise_scale = args.glm_noise_scale
    glm_groups = args.glm_groups
    theta_scale = args.theta_scale
    glm_link_str = args.glm_link
    tau_constant = args.tau_constant
    goptimal_iters = args.goptimal_iters

    glm_link_func = logistic if glm_link_str == 'logistic' else linear_link

    if use_sushi_data:
        K = 10  # Sushi dataset has K=10
        print(f"--- Running on Sushi Dataset (U={U} clusters, K={K}, T={T}) ---")
        mode_str = 'GLM' if sushi_as_glm else 'Tabular'
        print(f"Mode: {mode_str}")

        # Load data once. Clustering is deterministic, so it's the same for all runs.
        pref_matrices, _, phi_sushi = load_sushi_data(n_clusters=U, glm_features=sushi_as_glm, link=glm_link_func if sushi_as_glm else None)
        s_true = borda_score_from_preferences(pref_matrices)

        rng_seeder = np.random.default_rng(seed=42)
        seeds = rng_seeder.integers(0, 100000, size=run_counts)

        if sushi_as_glm:
            if phi_sushi is None:
                print("Error: --sushi_as_glm requires feature extraction from sushi data, but phi is None.", file=sys.stderr)
                sys.exit(1)
            phi_glm = phi_sushi
            d = phi_glm.shape[2]

            glm_regret = np.zeros((run_counts, T))
            glm_scores_1 = np.zeros((run_counts, T, U))
            glm_scores_2 = np.zeros((run_counts, T, U))

            glm_winner_regret = np.zeros((run_counts, T))
            glm_winner_scores_1 = np.zeros((run_counts, T, U))
            glm_winner_scores_2 = np.zeros((run_counts, T, U))

            glm_util_regret = np.zeros((run_counts, T))
            glm_util_scores_1 = np.zeros((run_counts, T, U))
            glm_util_scores_2 = np.zeros((run_counts, T, U))

            glm_uniform_winners_regret = np.zeros((run_counts, T))
            glm_uniform_winners_scores_1 = np.zeros((run_counts, T, U))
            glm_uniform_winners_scores_2 = np.zeros((run_counts, T, U))

            for i in range(run_counts):
                s = seeds[i]
                print(f"  Running seed {s} ({i+1}/{run_counts})...", flush=True)
                rng = np.random.default_rng(seed=s)

                print("  Running BETC-GLM (NSW)...", flush=True)
                glm_results = betc_glm(P_users=pref_matrices, s_true=s_true, phi=phi_glm, T=T, rng=rng, tau_constant=tau_constant, iters_fw=iters_fw, link=glm_link_func, goptimal_iters=goptimal_iters)
                glm_regret[i], _, glm_scores_1[i], glm_scores_2[i] = glm_results

                print("  Running BETC-GLM-Winner...", flush=True)
                glm_winner_results = betc_glm_winner(P_users=pref_matrices, s_true=s_true, phi=phi_glm, T=T, rng=rng, tau_constant=tau_constant, iters_fw=iters_fw, link=glm_link_func, goptimal_iters=goptimal_iters)
                glm_winner_regret[i], _, glm_winner_scores_1[i], glm_winner_scores_2[i] = glm_winner_results

                print("  Running BETC-GLM (Utilitarian)...", flush=True)
                glm_util_results = betc_glm(P_users=pref_matrices, s_true=s_true, phi=phi_glm, T=T, rng=rng, tau_constant=tau_constant, iters_fw=iters_fw, link=glm_link_func, goptimal_iters=goptimal_iters, utilitarian=True)
                glm_util_regret[i], _, glm_util_scores_1[i], glm_util_scores_2[i] = glm_util_results

                print("  Running BETC-GLM (Uniform over Winners)...", flush=True)
                glm_unif_win_results = betc_glm_uniform_winners(P_users=pref_matrices, s_true=s_true, phi=phi_glm, T=T, rng=rng, tau_constant=tau_constant, iters_fw=iters_fw, link=glm_link_func, goptimal_iters=goptimal_iters)
                glm_uniform_winners_regret[i], _, glm_uniform_winners_scores_1[i], glm_uniform_winners_scores_2[i] = glm_unif_win_results

            output_dir = os.path.join(base_directory, f"{U}_{K}_{T}_sushi_glm")
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nSaving results to {output_dir}...")

            np.save(os.path.join(output_dir, "glm_regret.npy"), glm_regret)
            np.save(os.path.join(output_dir, "glm_scores_1.npy"), glm_scores_1)
            np.save(os.path.join(output_dir, "glm_scores_2.npy"), glm_scores_2)

            np.save(os.path.join(output_dir, "glm_winner_regret.npy"), glm_winner_regret)
            np.save(os.path.join(output_dir, "glm_winner_scores_1.npy"), glm_winner_scores_1)
            np.save(os.path.join(output_dir, "glm_winner_scores_2.npy"), glm_winner_scores_2)

            np.save(os.path.join(output_dir, "glm_util_regret.npy"), glm_util_regret)
            np.save(os.path.join(output_dir, "glm_util_scores_1.npy"), glm_util_scores_1)
            np.save(os.path.join(output_dir, "glm_util_scores_2.npy"), glm_util_scores_2)

            np.save(os.path.join(output_dir, "glm_uniform_winners_regret.npy"), glm_uniform_winners_regret)
            np.save(os.path.join(output_dir, "glm_uniform_winners_scores_1.npy"), glm_uniform_winners_scores_1)
            np.save(os.path.join(output_dir, "glm_uniform_winners_scores_2.npy"), glm_uniform_winners_scores_2)

            with open(os.path.join(output_dir, "params.txt"), "w") as f:
                f.write(f"Experiment Type: Sushi Dataset (GLM Mode)\n")
                f.write(f"U={U} (clusters)\nK={K}\nT={T}\nrun_counts={run_counts}\n")
                f.write(f"d={d}\nglm_link={glm_link_str}\n")
                f.write(f"tau_constant={tau_constant}\ngoptimal_iters={goptimal_iters}\n")
                f.write(f"iters_fw={iters_fw}\n")

        else: # Tabular case
            etc_borda_regret = np.zeros((run_counts, T))
            etc_borda_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_scores_2 = np.zeros((run_counts, T, U))

            etc_borda_winner_regret = np.zeros((run_counts, T))
            etc_borda_winner_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_winner_scores_2 = np.zeros((run_counts, T, U))

            etc_borda_util_regret = np.zeros((run_counts, T))
            etc_borda_util_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_util_scores_2 = np.zeros((run_counts, T, U))

            etc_borda_uniform_winners_regret = np.zeros((run_counts, T))
            etc_borda_uniform_winners_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_uniform_winners_scores_2 = np.zeros((run_counts, T, U))

            for i in range(run_counts):
                s = seeds[i]
                print(f"  Running seed {s} ({i+1}/{run_counts})...", flush=True)
                rng = np.random.default_rng(seed=s)

                print("  Running Borda-ETC...", flush=True)
                etc_borda_results = explore_then_commit_borda(P_users=pref_matrices, s_true=s_true, T=T, rng=rng, iters_fw=iters_fw, exploration_frac=exploration_frac)
                etc_borda_regret[i], _, etc_borda_scores_1[i], etc_borda_scores_2[i] = etc_borda_results

                print("  Running Borda-Winner-ETC...", flush=True)
                etc_borda_winner_results = explore_then_commit_borda_winner(P_users=pref_matrices, s_true=s_true, T=T, rng=rng, iters_fw=iters_fw, exploration_frac=exploration_frac)
                etc_borda_winner_regret[i], _, etc_borda_winner_scores_1[i], etc_borda_winner_scores_2[i] = etc_borda_winner_results

                print("  Running Borda-ETC (Utilitarian)...", flush=True)
                etc_borda_util_results = explore_then_commit_borda(P_users=pref_matrices, s_true=s_true, T=T, rng=rng, iters_fw=iters_fw, exploration_frac=exploration_frac, utilitarian=True)
                etc_borda_util_regret[i], _, etc_borda_util_scores_1[i], etc_borda_util_scores_2[i] = etc_borda_util_results

                print("  Running Borda-ETC (Uniform over Winners)...", flush=True)
                etc_borda_unif_win_results = explore_then_commit_borda_uniform_winners(P_users=pref_matrices, s_true=s_true, T=T, rng=rng, iters_fw=iters_fw, exploration_frac=exploration_frac)
                etc_borda_uniform_winners_regret[i], _, etc_borda_uniform_winners_scores_1[i], etc_borda_uniform_winners_scores_2[i] = etc_borda_unif_win_results

            output_dir = os.path.join(base_directory, f"{U}_{K}_{T}_sushi_tabular")
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nSaving results to {output_dir}...")

            np.save(os.path.join(output_dir, "etc_regret.npy"), etc_borda_regret)
            np.save(os.path.join(output_dir, "etc_scores_1.npy"), etc_borda_scores_1)
            np.save(os.path.join(output_dir, "etc_scores_2.npy"), etc_borda_scores_2)

            np.save(os.path.join(output_dir, "etc_winner_regret.npy"), etc_borda_winner_regret)
            np.save(os.path.join(output_dir, "etc_winner_scores_1.npy"), etc_borda_winner_scores_1)
            np.save(os.path.join(output_dir, "etc_winner_scores_2.npy"), etc_borda_winner_scores_2)

            np.save(os.path.join(output_dir, "etc_util_regret.npy"), etc_borda_util_regret)
            np.save(os.path.join(output_dir, "etc_util_scores_1.npy"), etc_borda_util_scores_1)
            np.save(os.path.join(output_dir, "etc_util_scores_2.npy"), etc_borda_util_scores_2)

            np.save(os.path.join(output_dir, "etc_uniform_winners_regret.npy"), etc_borda_uniform_winners_regret)
            np.save(os.path.join(output_dir, "etc_uniform_winners_scores_1.npy"), etc_borda_uniform_winners_scores_1)
            np.save(os.path.join(output_dir, "etc_uniform_winners_scores_2.npy"), etc_borda_uniform_winners_scores_2)

            with open(os.path.join(output_dir, "params.txt"), "w") as f:
                f.write(f"Experiment Type: Sushi Dataset (Tabular Mode)\n")
                f.write(f"U={U} (clusters)\nK={K}\nT={T}\nrun_counts={run_counts}\n")
                f.write(f"exploration_frac={exploration_frac}\n")
                f.write(f"iters_fw={iters_fw}\n")
        
        print("Done.")

    else:
        rng_seeder = np.random.default_rng(seed=42) # Default case for random/conflicting GLM

        seeds = rng_seeder.integers(0, 100000, size=run_counts)
        
        # Conditional storage based on instance type
        if use_glm_instance or use_conflicting_glm_instance:
            glm_regret = np.zeros((run_counts, T))
            glm_ex_steps = np.zeros(run_counts)
            glm_scores_1 = np.zeros((run_counts, T, U))
            glm_scores_2 = np.zeros((run_counts, T, U))

            glm_winner_regret = np.zeros((run_counts, T))
            glm_winner_ex_steps = np.zeros(run_counts)
            glm_winner_scores_1 = np.zeros((run_counts, T, U))
            glm_winner_scores_2 = np.zeros((run_counts, T, U))

            glm_util_regret = np.zeros((run_counts, T))
            glm_util_scores_1 = np.zeros((run_counts, T, U))
            glm_util_scores_2 = np.zeros((run_counts, T, U))

            glm_uniform_winners_regret = np.zeros((run_counts, T))
            glm_uniform_winners_ex_steps = np.zeros(run_counts)
            glm_uniform_winners_scores_1 = np.zeros((run_counts, T, U))
            glm_uniform_winners_scores_2 = np.zeros((run_counts, T, U))

        else:
            # Storage for Borda-ETC
            etc_borda_regret = np.zeros((run_counts, T))
            etc_borda_ex_steps = np.zeros(run_counts)
            etc_borda_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_scores_2 = np.zeros((run_counts, T, U))

            # Storage for Borda-Winner-ETC
            etc_borda_winner_regret = np.zeros((run_counts, T))
            etc_borda_winner_ex_steps = np.zeros(run_counts)
            etc_borda_winner_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_winner_scores_2 = np.zeros((run_counts, T, U))

            etc_borda_util_regret = np.zeros((run_counts, T))
            etc_borda_util_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_util_scores_2 = np.zeros((run_counts, T, U))

            etc_borda_uniform_winners_regret = np.zeros((run_counts, T))
            etc_borda_uniform_winners_ex_steps = np.zeros(run_counts)
            etc_borda_uniform_winners_scores_1 = np.zeros((run_counts, T, U))
            etc_borda_uniform_winners_scores_2 = np.zeros((run_counts, T, U))

        for i in range(run_counts):
            s = seeds[i]
            print(f"Running seed {s} ({i+1}/{run_counts})...", flush=True)
            rng = np.random.default_rng(seed=s)
            
            if use_glm_instance or use_conflicting_glm_instance:
                if use_conflicting_glm_instance:
                    print("Generating conflicting GLM instance...", flush=True)
                    from utils import generate_conflicting_glm_instance
                    pref_matrices, phi_glm = generate_conflicting_glm_instance(
                        K=K, U=U, d=d, rng=rng, theta_scale=theta_scale, link=glm_link_func,
                        noise_scale=glm_noise_scale, num_groups=glm_groups
                    )
                else: # Standard GLM
                    print("Generating GLM instance...", flush=True)
                    pref_matrices, phi_glm = generate_random_glm_instance(
                        K=K, U=U, d=d, rng=rng, theta_scale=theta_scale, link=glm_link_func)
                
                s_true = borda_score_from_preferences(pref_matrices)

                print("Running BETC-GLM (NSW)...", flush=True)
                glm_results = betc_glm(
                    P_users=pref_matrices,
                    s_true=s_true,
                    phi=phi_glm,
                    T=T,
                    rng=rng,
                    tau_constant=tau_constant,
                    iters_fw=iters_fw,
                    link=glm_link_func,
                    goptimal_iters=goptimal_iters,
                )
                glm_regret[i], glm_ex_steps[i], glm_scores_1[i], glm_scores_2[i] = glm_results

                print("Running BETC-GLM-Winner...", flush=True)
                glm_winner_results = betc_glm_winner(
                    P_users=pref_matrices,
                    s_true=s_true,
                    phi=phi_glm,
                    T=T,
                    rng=rng,
                    tau_constant=tau_constant,
                    iters_fw=iters_fw,
                    link=glm_link_func,
                    goptimal_iters=goptimal_iters,
                )
                glm_winner_regret[i], glm_winner_ex_steps[i], glm_winner_scores_1[i], glm_winner_scores_2[i] = glm_winner_results

                print("Running BETC-GLM (Utilitarian)...", flush=True)
                glm_util_results = betc_glm(
                    P_users=pref_matrices,
                    s_true=s_true,
                    phi=phi_glm,
                    T=T,
                    rng=rng,
                    tau_constant=tau_constant,
                    iters_fw=iters_fw,
                    link=glm_link_func,
                    goptimal_iters=goptimal_iters,
                    utilitarian=True,
                )
                glm_util_regret[i], _, glm_util_scores_1[i], glm_util_scores_2[i] = glm_util_results

                print("Running BETC-GLM (Uniform over Winners)...", flush=True)
                glm_unif_win_results = betc_glm_uniform_winners(
                    P_users=pref_matrices,
                    s_true=s_true,
                    phi=phi_glm,
                    T=T,
                    rng=rng,
                    tau_constant=tau_constant,
                    iters_fw=iters_fw,
                    link=glm_link_func,
                    goptimal_iters=goptimal_iters,
                )
                glm_uniform_winners_regret[i], glm_uniform_winners_ex_steps[i], glm_uniform_winners_scores_1[i], glm_uniform_winners_scores_2[i] = glm_unif_win_results
            else:
                print("Generating random user preferences...", flush=True)
                pref_matrices = generate_user_prefs(U=U, K=K, rng=rng)
                s_true = borda_score_from_preferences(pref_matrices)

                # Run Borda-ETC
                print("Running Borda-ETC...", flush=True)
                etc_borda_results = explore_then_commit_borda(
                    P_users=pref_matrices,
                    s_true=s_true,
                    T=T,
                    rng=rng,
                    iters_fw=iters_fw,
                    exploration_frac=exploration_frac,
                )
                etc_borda_regret[i], etc_borda_ex_steps[i], etc_borda_scores_1[i], etc_borda_scores_2[i] = etc_borda_results

                # Run Borda-Winner-ETC
                print("Running Borda-Winner-ETC...", flush=True)
                etc_borda_winner_results = explore_then_commit_borda_winner(
                    P_users=pref_matrices,
                    s_true=s_true,
                    T=T,
                    rng=rng,
                    iters_fw=iters_fw,
                    exploration_frac=exploration_frac,
                )
                etc_borda_winner_regret[i], etc_borda_winner_ex_steps[i], etc_borda_winner_scores_1[i], etc_borda_winner_scores_2[i] = etc_borda_winner_results

                # Run Borda-ETC (Utilitarian)
                print("Running Borda-ETC (Utilitarian)...", flush=True)
                etc_borda_util_results = explore_then_commit_borda(
                    P_users=pref_matrices,
                    s_true=s_true,
                    T=T,
                    rng=rng,
                    iters_fw=iters_fw,
                    exploration_frac=exploration_frac,
                    utilitarian=True,
                )
                etc_borda_util_regret[i], _, etc_borda_util_scores_1[i], etc_borda_util_scores_2[i] = etc_borda_util_results

                # Run Borda-ETC (Uniform over Winners)
                print("Running Borda-ETC (Uniform over Winners)...", flush=True)
                etc_borda_unif_win_results = explore_then_commit_borda_uniform_winners(
                    P_users=pref_matrices,
                    s_true=s_true,
                    T=T,
                    rng=rng,
                    iters_fw=iters_fw,
                    exploration_frac=exploration_frac,
                )
                etc_borda_uniform_winners_regret[i], etc_borda_uniform_winners_ex_steps[i], etc_borda_uniform_winners_scores_1[i], etc_borda_uniform_winners_scores_2[i] = etc_borda_unif_win_results
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if use_glm_instance or use_conflicting_glm_instance:
            if use_conflicting_glm_instance:
                output_dir = f"{base_directory}/{U}_{K}_{T}_{d}_conflicting_glm"
            else:
                output_dir = f"{base_directory}/{U}_{K}_{T}_{d}_glm"
        else:
            output_dir = f"{base_directory}/{U}_{K}_{T}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to {output_dir}...")

        if use_glm_instance or use_conflicting_glm_instance:
            np.save(os.path.join(output_dir, "glm_regret.npy"), glm_regret)
            np.save(os.path.join(output_dir, "glm_scores_1.npy"), glm_scores_1)
            np.save(os.path.join(output_dir, "glm_scores_2.npy"), glm_scores_2)

            np.save(os.path.join(output_dir, "glm_winner_regret.npy"), glm_winner_regret)
            np.save(os.path.join(output_dir, "glm_winner_scores_1.npy"), glm_winner_scores_1)
            np.save(os.path.join(output_dir, "glm_winner_scores_2.npy"), glm_winner_scores_2)

            np.save(os.path.join(output_dir, "glm_util_regret.npy"), glm_util_regret)
            np.save(os.path.join(output_dir, "glm_util_scores_1.npy"), glm_util_scores_1)
            np.save(os.path.join(output_dir, "glm_util_scores_2.npy"), glm_util_scores_2)

            np.save(os.path.join(output_dir, "glm_uniform_winners_regret.npy"), glm_uniform_winners_regret)
            np.save(os.path.join(output_dir, "glm_uniform_winners_scores_1.npy"), glm_uniform_winners_scores_1)
            np.save(os.path.join(output_dir, "glm_uniform_winners_scores_2.npy"), glm_uniform_winners_scores_2)

            # Save parameters for reference
            with open(os.path.join(output_dir, "params.txt"), "w") as f:
                f.write(f"U={U}\nK={K}\nT={T}\nrun_counts={run_counts}\n")
                f.write(f"use_glm_instance={use_glm_instance or use_conflicting_glm_instance}\n")
                f.write(f"conflicting_instance={use_conflicting_glm_instance}\n")
                f.write(f"d={d}\ntheta_scale={theta_scale}\nglm_link={glm_link_str}\n")
                f.write(f"tau_constant={tau_constant}\ngoptimal_iters={goptimal_iters}\n")
                if use_conflicting_glm_instance:
                    f.write(f"glm_noise_scale={glm_noise_scale}\nglm_groups={glm_groups}\n")
                f.write(f"iters_fw={iters_fw}\n")
        else:
            np.save(os.path.join(output_dir, "etc_regret.npy"), etc_borda_regret)
            np.save(os.path.join(output_dir, "etc_scores_1.npy"), etc_borda_scores_1)
            np.save(os.path.join(output_dir, "etc_scores_2.npy"), etc_borda_scores_2)

            np.save(os.path.join(output_dir, "etc_winner_regret.npy"), etc_borda_winner_regret)
            np.save(os.path.join(output_dir, "etc_winner_scores_1.npy"), etc_borda_winner_scores_1)
            np.save(os.path.join(output_dir, "etc_winner_scores_2.npy"), etc_borda_winner_scores_2)

            np.save(os.path.join(output_dir, "etc_util_regret.npy"), etc_borda_util_regret)
            np.save(os.path.join(output_dir, "etc_util_scores_1.npy"), etc_borda_util_scores_1)
            np.save(os.path.join(output_dir, "etc_util_scores_2.npy"), etc_borda_util_scores_2)

            np.save(os.path.join(output_dir, "etc_uniform_winners_regret.npy"), etc_borda_uniform_winners_regret)
            np.save(os.path.join(output_dir, "etc_uniform_winners_scores_1.npy"), etc_borda_uniform_winners_scores_1)
            np.save(os.path.join(output_dir, "etc_uniform_winners_scores_2.npy"), etc_borda_uniform_winners_scores_2)

            # Save parameters for reference
            with open(os.path.join(output_dir, "params.txt"), "w") as f:
                f.write(f"U={U}\nK={K}\nT={T}\nrun_counts={run_counts}\n")
                f.write(f"exploration_frac={exploration_frac}\n")
                f.write(f"iters_fw={iters_fw}\n")
                f.write(f"use_glm_instance={use_glm_instance}\n")