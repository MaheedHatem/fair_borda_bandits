# Borda-Based Fair Multi-User Dueling Bandit in Tabular and Generalized Linear Settings

Official implementation of the paper:

> **Borda-Based Fair Multi-User Dueling Bandit in Tabular and Generalized Linear Settings**
> Maheed H. Ahmed and Mahsa Ghasemi
> *NeurIPS 2026*
>
> Paper link coming soon.

The code studies dueling bandits with multiple users, where each user has their own preference matrix over `K` arms. The learner aims to maximise the **Nash Social Welfare (NSW)** of the users' normalised Borda scores. Explore-then-commit algorithms are provided for the tabular setting and for the generalised linear model (GLM) setting, compared against utilitarian and winner-based baselines.

## Installation

Tested with Python 3.11.

```bash
git clone git@github.com:MaheedHatem/fair_borda_bandits.git
cd fair_borda_bandits
pip install -r requirements.txt
```

The plots are rendered with LaTeX (`text.usetex: True` in `plot_results.py`), so a LaTeX distribution must be installed (for example TeX Live) to run the plotting scripts.

### Sushi dataset (optional)

The real-data experiments use the [Sushi Preference Dataset](https://www.kamishima.net/sushi/). Download it and extract it so that the following files exist:

```
data/sushi3-2016/sushi3a.5000.10.order   # rankings (required for all Sushi runs)
data/sushi3-2016/sushi3.idata            # item features (required for --sushi_as_glm)
```

## Running experiments

All experiments are launched through `main.py`. Each run executes four algorithms on the same instance for every seed:

| Algorithm | Tabular | GLM | Commit policy |
|---|---|---|---|
| Fair-BETC (ours) | `explore_then_commit_borda` | `betc_glm` | NSW-optimal mixed policy over arms |
| Utilitarian | `explore_then_commit_borda(utilitarian=True)` | `betc_glm(utilitarian=True)` | Maximises the average normalised Borda score |
| Winner-Commit | `explore_then_commit_borda_winner` | `betc_glm_winner` | Arm with the highest total Borda score |
| Uniform-Winners | `explore_then_commit_borda_uniform_winners` | `betc_glm_uniform_winners` | Uniform over each user's Borda winner |

### Random tabular instances

```bash
python main.py --U 5 --K 10 --T 200000 --run_counts 30 --base_directory ./results
```

### Random GLM instances

```bash
python main.py --use_glm_instance --U 5 --K 10 --T 100000 --d 6 --run_counts 30 --base_directory ./results
```

Use `--use_conflicting_glm_instance` instead to split users into opposing preference groups (`--glm_groups`, `--glm_noise_scale`).

### Sushi dataset

```bash
# Tabular: empirical preference matrices of U user clusters
python main.py --use_sushi_data --U 5 --T 100000 --run_counts 30 --base_directory ./results

# GLM: cluster preferences projected onto a logistic GLM over item features
python main.py --use_sushi_data --sushi_as_glm --U 5 --T 100000 --run_counts 10 --base_directory ./results
```

For Sushi data, `--U` is the number of user clusters and `K` is fixed to 10.

### Main options

| Flag | Default | Description |
|---|---|---|
| `--U`, `--K`, `--T` | 8, 10, 10000 | Users, arms, horizon |
| `--run_counts` | 10 | Number of seeds |
| `--d` | 5 | Feature dimension for GLM instances |
| `--glm_link` | `logistic` | `logistic` or `linear` |
| `--exploration_frac` | 0.25 | Constant in the tabular exploration length |
| `--tau_constant` | 100 | Constant in the GLM random-exploration length |
| `--goptimal_iters` | 500 | Frank-Wolfe iterations for the G-optimal design |
| `--iters_fw` | 300 | Frank-Wolfe iterations for the NSW policy |
| `--base_directory` | `./results_borda/` | Where result folders are written |

Run `python main.py --help` for the full list.

### Output

Each run writes a folder in `--base_directory` named after its settings: `{U}_{K}_{T}` for tabular runs, `{U}_{K}_{T}_{d}_glm` for GLM runs, `{U}_{K}_{T}_{d}_conflicting_glm` for conflicting-group runs, and `{U}_10_{T}_sushi_{tabular,glm}` for Sushi runs. Each folder contains `params.txt` and, for every algorithm, `*_regret.npy` (cumulative NSW regret) and `*_scores_{1,2}.npy` (per-user scores of the two arms played each round).

## Plotting

```bash
# All result folders in a directory
python plot_all.py --base_directory ./results --out_directory ./Figures

# A single result folder
python plot_results.py ./results/5_10_200000 --output_dir ./Figures --suffix 5_10_200000
```

For each folder this produces PDFs of cumulative regret, minimum / NSW / utilitarian user welfare and the Gini coefficient (mean with 95% confidence intervals), plus a `metrics_table_*.txt` with the final-round values.

## Regret scaling with the number of users

These scripts run the algorithms on hard instance families over a range of `U`, fit the log-log slope of the final regret, and compare it to the reference slopes 1/3 and 2/3:

```bash
# Tabular Fair-BETC on the hard tabular instance
python u_scaling_check.py --U_values 4,8,16,32,64,128 --K 11 --T 200000 --run_counts 20

# Fair-BETC-GLM on the hard linear-link GLM instance (requires K >= 2^d)
python u_scaling_check_glm_hard.py --U_values 4,8,16,32,64,128 --K 16 --d 3 --T 600000 --run_counts 20
```

Results (CSV and PDFs) are written to `experiments/results/u_scaling/` and `experiments/results/u_scaling_glm_hard/` by default (`--output_dir` changes this).

## Code structure

```
main.py                      Experiment entry point
algorithms.py                Tabular and GLM explore-then-commit algorithms, GLM MLE
utils.py                     Instance generators, Sushi loader, Borda scores, Frank-Wolfe solvers
plot_results.py              Plots and metric tables for one result folder
plot_all.py                  Runs plot_results.py on every result folder in a directory
u_scaling_check.py           Regret-vs-U experiment, tabular hard instance
u_scaling_check_glm_hard.py  Regret-vs-U experiment, GLM hard instance
```

## Citation

The BibTeX entry will be added once the NeurIPS 2026 proceedings are available.
