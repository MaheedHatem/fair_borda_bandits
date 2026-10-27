import numpy as np
import math
from numpy.random import Generator
from typing import List, Tuple
from utils import frank_wolfe_goptimal, sample_duel, objective_and_grad, frank_wolfe_simplex, get_borda_scores
from typing import Callable, Optional
import scipy.optimize as opt
from utils import logistic, linear_link

def betc_glm(
    P_users: np.ndarray,
    s_true: np.ndarray,
    phi: np.ndarray,
    T: int,
    rng: Generator,
    tau_constant: float = 100.0,
    iters_fw: int = 300,
    link: Callable = logistic,
    goptimal_iters: int = 500,
    utilitarian: bool = False,
) -> Tuple:
    U, K, _ = P_users.shape
    d = phi.shape[2]

    tau = int(np.ceil(tau_constant * (d + np.log(U * T))))
    eps = d ** (1/6) * T ** (-1/3) * U ** (-1/3)
    tau = min(tau, T)

    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)
    scores_1 = []
    scores_2 = []

    phi_seq = []
    r_seq   = []

    def compute_regret_step(x_1: np.ndarray, x_2: np.ndarray, t_idx: int):
        """
        Compute regret for a round where arm 1 is drawn from policy x_1
        and arm 2 from policy x_2. Regret is based on the expected NSW
        value of the mixed policy, not on the realised arm samples.
        """
        fx_val_1, _ = objective_and_grad(x_1, s_true)
        fx_val_2, _ = objective_and_grad(x_2, s_true)
        fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
        prev = cumulative_regret[t_idx - 1] if t_idx > 0 else 0.0
        cumulative_regret[t_idx] = prev + (f_star - fx)

    def query(i, j, t_idx, x_1: np.ndarray, x_2: np.ndarray):
        """Query pair (i, j), record GLM feedback, compute policy-based regret."""
        outcomes = sample_duel(P_users, i, j, rng)     # shape (U,)
        phi_seq.append(phi[i, j])
        r_seq.append(outcomes)

        scores_1.append(s_true[:, i])
        scores_2.append(s_true[:, j])

        compute_regret_step(x_1, x_2, t_idx)

    # ---- Phase 1: Pure random exploration --------------------------------
    # Policy for both arms is uniform over all K arms
    x_uniform = np.ones(K) / K

    t = 0
    for t in range(min(tau, T)):
        i = rng.integers(K)
        j = rng.integers(K)
        query(i, j, t, x_uniform, x_uniform)

    t += 1

    if t >= T:
        print("Warning: pure exploration filled the entire budget T.")
        return (cumulative_regret, t, np.array(scores_1), np.array(scores_2))

    # ---- G-optimal design ------------------------------------------------
    pi = frank_wolfe_goptimal(phi, n_iter=goptimal_iters)

    N_pairs = np.ceil(d * pi / (eps ** 2)).astype(int)
    designed_pairs = [(i, j, N_pairs[i, j])
                      for i in range(K) for j in range(K)
                      if N_pairs[i, j] > 0]

    # ---- Phase 2: Designed exploration -----------------------------------
    # Policy for each query is deterministic on the prescribed arms i and j
    N_used = 0
    for (i, j, n_ij) in designed_pairs:
        x_i = np.zeros(K); x_i[i] = 1.0
        x_j = np.zeros(K); x_j[j] = 1.0
        for _ in range(n_ij):
            if t >= T:
                break
            query(i, j, t, x_i, x_j)
            t += 1
            N_used += 1
        if t >= T:
            break

    exploration_steps = t

    if t >= T:
        print("Warning: designed exploration filled the entire budget T.")
        return (cumulative_regret, exploration_steps,
                np.array(scores_1), np.array(scores_2))

    # ---- MLE estimation per user -----------------------------------------
    phi_arr = np.array(phi_seq)     # (n, d)
    r_arr   = np.array(r_seq)       # (n, U)

    borda_scores_est = np.zeros((U, K)) # Will store mean Borda scores
    for u in range(U):
        theta_u = mle_glm(phi_arr, r_arr[:, u], link=link, d=d)
        # Estimate preference matrix P_hat for user u
        P_hat_u = link(np.einsum('ijd,d->ij', phi, theta_u))
        # Borda score is sum_{j!=i} P(i,j). We use the mean over j!=i.
        sum_borda = np.sum(P_hat_u, axis=1) - 0.5 # P_hat(i,i) is link(0)=0.5
        if K > 1:
            borda_scores_est[u, :] = sum_borda / (K - 1)

    scores_est = np.zeros((U, K))
    for u in range(U):
        max_borda = np.max(borda_scores_est[u, :])
        scores_est[u, :] = borda_scores_est[u, :] / max(0.5, max_borda)
    assert not np.any(scores_est < 0.0) and not np.any(scores_est > 1.0), \
        "Estimated scores should be in [0, 1]"

    # ---- Commit: NSW-optimal policy --------------------------------------
    x_hat, _ = frank_wolfe_simplex(scores_est, K, iters=iters_fw, utilitarian=utilitarian)

    fx_val, _ = objective_and_grad(x_hat, s_true)
    fx = np.exp(fx_val)

    cumulative_regret[t:] = (
        cumulative_regret[t - 1] + np.cumsum(np.full(T - t, f_star - fx))
    )

    for _ in range(t, T):
        i, j = rng.choice(K, size=2, p=x_hat)
        scores_1.append(s_true[:, i])
        scores_2.append(s_true[:, j])

    return (cumulative_regret, exploration_steps,
            np.array(scores_1), np.array(scores_2))

def betc_glm_winner(
    P_users: np.ndarray,
    s_true: np.ndarray,
    phi: np.ndarray,
    T: int,
    rng: Generator,
    tau_constant: float = 100.0,
    iters_fw: int = 300,
    link: Callable = logistic,
    goptimal_iters: int = 500,
) -> Tuple:
    """
    Explore-then-commit algorithm with GLM, committing to the Borda winner.
    Exploration: Same as BETC-GLM (random + G-optimal design).
    Estimation: MLE for GLM parameters, then estimate Borda scores.
    Commit: Find the arm with the highest total estimated Borda score and play it.
    """
    U, K, _ = P_users.shape
    d = phi.shape[2]

    tau = int(np.ceil(tau_constant * (d + np.log(U * T))))
    eps = d ** (1/6) * T ** (-1/3) * U ** (-1/3)
    tau = min(tau, T)

    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)
    scores_1 = []
    scores_2 = []

    phi_seq = []
    r_seq   = []

    def compute_regret_step(x_1: np.ndarray, x_2: np.ndarray, t_idx: int):
        fx_val_1, _ = objective_and_grad(x_1, s_true)
        fx_val_2, _ = objective_and_grad(x_2, s_true)
        fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
        prev = cumulative_regret[t_idx - 1] if t_idx > 0 else 0.0
        cumulative_regret[t_idx] = prev + (f_star - fx)

    def query(i, j, t_idx, x_1: np.ndarray, x_2: np.ndarray):
        outcomes = sample_duel(P_users, i, j, rng)
        phi_seq.append(phi[i, j])
        r_seq.append(outcomes)
        scores_1.append(s_true[:, i])
        scores_2.append(s_true[:, j])
        compute_regret_step(x_1, x_2, t_idx)

    x_uniform = np.ones(K) / K
    t = 0
    for t in range(min(tau, T)):
        i, j = rng.integers(K), rng.integers(K)
        query(i, j, t, x_uniform, x_uniform)

    t += 1

    if t >= T: return cumulative_regret, t, np.array(scores_1), np.array(scores_2)

    pi = frank_wolfe_goptimal(phi, n_iter=goptimal_iters)
    N_pairs = np.ceil(d * pi / (eps ** 2)).astype(int)
    designed_pairs = [(i, j, N_pairs[i, j]) for i in range(K) for j in range(K) if N_pairs[i, j] > 0]

    for (i, j, n_ij) in designed_pairs:
        x_i, x_j = np.zeros(K), np.zeros(K)
        x_i[i], x_j[j] = 1.0, 1.0
        for _ in range(n_ij):
            if t >= T: break
            query(i, j, t, x_i, x_j)
            t += 1
        if t >= T: break
    exploration_steps = t

    if t >= T: return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)

    phi_arr, r_arr = np.array(phi_seq), np.array(r_seq)
    borda_scores_est = np.zeros((U, K))
    for u in range(U):
        theta_u = mle_glm(phi_arr, r_arr[:, u], link=link, d=d)
        # Estimate preference matrix P_hat for user u
        P_hat_u = link(np.einsum('ijd,d->ij', phi, theta_u))
        sum_borda = np.sum(P_hat_u, axis=1) - 0.5 # P_hat(i,i) is link(0)=0.5
        if K > 1:
            borda_scores_est[u, :] = sum_borda / (K - 1)

    total_borda_scores_est = np.sum(borda_scores_est, axis=0) # Sum of mean Borda scores over users
    winner_arm = np.argmax(total_borda_scores_est)
    x_hat = np.zeros(K); x_hat[winner_arm] = 1.0

    fx_val, _ = objective_and_grad(x_hat, s_true)
    fx = np.exp(fx_val)

    cumulative_regret[t:] = cumulative_regret[t - 1] + np.cumsum(np.full(T - t, f_star - fx))

    for _ in range(t, T):
        scores_1.append(s_true[:, winner_arm])
        scores_2.append(s_true[:, winner_arm])

    return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)

def betc_glm_uniform_winners(
    P_users: np.ndarray,
    s_true: np.ndarray,
    phi: np.ndarray,
    T: int,
    rng: Generator,
    tau_constant: float = 100.0,
    iters_fw: int = 300,
    link: Callable = logistic,
    goptimal_iters: int = 500,
) -> Tuple:
    """
    Explore-then-commit algorithm with GLM, committing to a uniform policy over per-user Borda winners.
    """
    U, K, _ = P_users.shape
    d = phi.shape[2]

    tau = int(np.ceil(tau_constant * (d + np.log(U * T))))
    eps = d ** (1/6) * T ** (-1/3) * U ** (-1/3)
    tau = min(tau, T)

    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)
    scores_1 = []
    scores_2 = []

    phi_seq = []
    r_seq   = []

    def compute_regret_step(x_1: np.ndarray, x_2: np.ndarray, t_idx: int):
        fx_val_1, _ = objective_and_grad(x_1, s_true)
        fx_val_2, _ = objective_and_grad(x_2, s_true)
        fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
        prev = cumulative_regret[t_idx - 1] if t_idx > 0 else 0.0
        cumulative_regret[t_idx] = prev + (f_star - fx)

    def query(i, j, t_idx, x_1: np.ndarray, x_2: np.ndarray):
        outcomes = sample_duel(P_users, i, j, rng)
        phi_seq.append(phi[i, j])
        r_seq.append(outcomes)
        scores_1.append(s_true[:, i])
        scores_2.append(s_true[:, j])
        compute_regret_step(x_1, x_2, t_idx)

    x_uniform = np.ones(K) / K
    t = 0
    for t in range(min(tau, T)):
        i, j = rng.integers(K), rng.integers(K)
        query(i, j, t, x_uniform, x_uniform)

    t += 1

    if t >= T: return cumulative_regret, t, np.array(scores_1), np.array(scores_2)

    pi = frank_wolfe_goptimal(phi, n_iter=goptimal_iters)
    N_pairs = np.ceil(d * pi / (eps ** 2)).astype(int)
    designed_pairs = [(i, j, N_pairs[i, j]) for i in range(K) for j in range(K) if N_pairs[i, j] > 0]

    for (i, j, n_ij) in designed_pairs:
        x_i, x_j = np.zeros(K), np.zeros(K)
        x_i[i], x_j[j] = 1.0, 1.0
        for _ in range(n_ij):
            if t >= T: break
            query(i, j, t, x_i, x_j)
            t += 1
        if t >= T: break
    exploration_steps = t

    if t >= T: return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)

    phi_arr, r_arr = np.array(phi_seq), np.array(r_seq)
    borda_scores_est = np.zeros((U, K))
    for u in range(U):
        theta_u = mle_glm(phi_arr, r_arr[:, u], link=link, d=d)
        P_hat_u = np.array([[link(phi[i, j] @ theta_u) for j in range(K)] for i in range(K)])
        borda_scores_est[u, :] = get_borda_scores(P_hat_u)

    user_winners = np.argmax(borda_scores_est, axis=1)
    x_hat = np.zeros(K)
    unique_winners, counts = np.unique(user_winners, return_counts=True)
    x_hat[unique_winners] = counts / U

    fx_val, _ = objective_and_grad(x_hat, s_true)
    fx = np.exp(fx_val)

    cumulative_regret[t:] = cumulative_regret[t - 1] + np.cumsum(np.full(T - t, f_star - fx))

    for _ in range(t, T):
        i, j = rng.choice(K, size=2, p=x_hat)
        scores_1.append(s_true[:, i])
        scores_2.append(s_true[:, j])

    return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)

def mle_glm(
    phi_seq: np.ndarray,    # (n, d)  context vectors of observed pairs
    r_seq: np.ndarray,      # (n,)    binary feedback (0 or 1)
    link: Callable = logistic,
    d: Optional[int] = None,
    theta_init: Optional[np.ndarray] = None,
    n_iter: int = 200,
) -> np.ndarray:
    """
    Solve the MLE problem (Equation 2) for a generalised linear model.
 
    Minimises  -sum_t [ r_t * phi_t^T theta - m(phi_t^T theta) ]
    where m'(·) = mu(·).
 
    For the logistic link:  m(x) = log(1 + exp(x)).
    For the linear  link:   closed-form solution is used directly.
    """
    n, d_obs = phi_seq.shape
    if d is None:
        d = d_obs
    if theta_init is None:
        theta_init = np.zeros(d)
 
    if link is linear_link:
        # Closed form: theta = V^{-1} sum_t (r_t - 0.5) phi_t
        V = phi_seq.T @ phi_seq + 1e-8 * np.eye(d)
        b = phi_seq.T @ (r_seq - 0.5)
        return np.linalg.solve(V, b)
 
    # General case: gradient descent on negative log-likelihood
    def neg_ll_and_grad(theta):
        eta   = phi_seq @ theta                      # (n,)
        mu    = link(eta)                            # (n,)
        mu    = np.clip(mu, 1e-9, 1 - 1e-9)
        loss  = -np.sum(r_seq * np.log(mu) + (1 - r_seq) * np.log(1 - mu))
        grad  = -phi_seq.T @ (r_seq - mu)
        return loss, grad
 
    res = opt.minimize(neg_ll_and_grad, theta_init, jac=True,
                       method="L-BFGS-B",
                       options={"maxiter": n_iter, "ftol": 1e-12})
    return res.x

def explore_then_commit_borda(
    P_users: np.ndarray,
    s_true: np.ndarray,
    T: int,
    rng: Generator,
    iters_fw: int = 300,
    exploration_frac: float = 0.25,
    utilitarian: bool = False,
) -> Tuple:
    """
    Explore-then-commit algorithm for learning a policy to maximize NSW of normalized Borda scores.
    Exploration: For each arm i, duel it L times against a uniformly random arm j.
    Scoring: Estimate Borda scores from exploration, normalize them, and use for optimization.
    """
    U, K, _ = P_users.shape

    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)
    scores_1 = []
    scores_2 = []

    # Phase 1: Exploration
    # Total exploration budget is exploration_frac * T.
    # Samples per arm is (exploration_frac * T) / K.
    L_expl = int(exploration_frac *(K ** (-2/3) * T ** (2/3) * U ** (2/3) * (math.log(U*K*T)) ** (1/3)) )
    
    wins_expl = np.zeros((U, K))
    counts_expl = np.zeros(K)
    
    t = 0
    for i in range(K):
        for _ in range(L_expl):
            if t >= T: break
            
            # Choose other arm uniformly at random
            j = rng.choice([arm for arm in range(K) if arm != i])
            
            # Sample duel
            outcomes = sample_duel(P_users, i, j, rng) # shape (U,)
            
            # Update wins
            wins_expl[:, i] += outcomes
            counts_expl[i] += 1
            
            # Calculate regret for this step
            x_hat_1 = np.zeros(K); x_hat_1[i] = 1.0
            x_hat_2 = np.ones(K)/(K-1); x_hat_2[i] = 0.0
            scores_1.append(s_true[:, i])
            scores_2.append(s_true[:, j])
            
            fx_val_1, _ = objective_and_grad(x_hat_1, s_true)
            fx_val_2, _ = objective_and_grad(x_hat_2, s_true)
            fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
            
            cumulative_regret[t] = (f_star - fx) + (cumulative_regret[t-1] if t > 0 else 0)
            t += 1
        if t >= T: break
    
    exploration_steps = t

    if t >= T:
        print("Warning: exploration filled the entire budget T.")
        return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)

    # Phase 2: Commit
    # Estimate scores
    borda_scores_est = np.zeros((U, K))
    for i in range(K):
        if counts_expl[i] > 0:
            avg_win_rate = wins_expl[:, i] / counts_expl[i]
            # avg_win_rate is an estimate of mean_{j!=i} P(i,j), which is the mean Borda score.
            borda_scores_est[:, i] = avg_win_rate

    scores_est = np.zeros((U, K))
    for u in range(U):
        max_borda = np.max(borda_scores_est[u, :])
        scores_est[u, :] = borda_scores_est[u, :] / max(0.5,max_borda)
    assert not np.any(scores_est < 0.0) and not np.any(scores_est > 1.0), "Estimated scores should be in [0,1]"

    x_hat, _ = frank_wolfe_simplex(scores_est, K, iters=iters_fw, utilitarian=utilitarian)
    
    fx_val, _ = objective_and_grad(x_hat, s_true)
    fx = np.exp(fx_val)
    
    cumulative_regret[t:] = cumulative_regret[t - 1] + np.cumsum(np.full(T-t, f_star - fx))

    # Fill scores for commit phase
    commit_scores_1 = []
    commit_scores_2 = []
    for _ in range(t, T):
        i, j = rng.choice(K, size=2, p=x_hat)
        commit_scores_1.append(s_true[:, i])
        commit_scores_2.append(s_true[:, j])
    
    scores_1.extend(commit_scores_1)
    scores_2.extend(commit_scores_2)

    return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)


def explore_then_commit_borda_uniform_winners(
    P_users: np.ndarray,
    s_true: np.ndarray,
    T: int,
    rng: Generator,
    iters_fw: int = 300,
    exploration_frac: float = 0.25,
) -> Tuple:
    """
    ETC algorithm that identifies the Borda winner for each user and commits to a uniform policy over these winners.
    """
    U, K, _ = P_users.shape

    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)
    scores_1 = []
    scores_2 = []

    L_expl = int(exploration_frac *(K ** (-2/3) * T ** (2/3) * U ** (2/3) * (math.log(U*K*T)) ** (1/3)) )
    
    wins_expl = np.zeros((U, K))
    counts_expl = np.zeros(K)
    
    t = 0
    for i in range(K):
        for _ in range(L_expl):
            if t >= T: break
            j = rng.choice([arm for arm in range(K) if arm != i])
            outcomes = sample_duel(P_users, i, j, rng)
            wins_expl[:, i] += outcomes
            counts_expl[i] += 1
            
            x_hat_1 = np.zeros(K); x_hat_1[i] = 1.0
            x_hat_2 = np.ones(K)/(K-1); x_hat_2[i] = 0.0
            scores_1.append(s_true[:, i])
            scores_2.append(s_true[:, j])
            
            fx_val_1, _ = objective_and_grad(x_hat_1, s_true)
            fx_val_2, _ = objective_and_grad(x_hat_2, s_true)
            fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
            
            cumulative_regret[t] = (f_star - fx) + (cumulative_regret[t-1] if t > 0 else 0)
            t += 1
        if t >= T: break
    
    exploration_steps = t

    if t >= T:
        return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)

    borda_scores_est = np.zeros((U, K))
    for i in range(K):
        if counts_expl[i] > 0:
            borda_scores_est[:, i] = wins_expl[:, i] / counts_expl[i]

    user_winners = np.argmax(borda_scores_est, axis=1)
    x_hat = np.zeros(K)
    unique_winners, counts = np.unique(user_winners, return_counts=True)
    x_hat[unique_winners] = counts / U
    
    fx_val, _ = objective_and_grad(x_hat, s_true)
    fx = np.exp(fx_val)
    
    cumulative_regret[t:] = cumulative_regret[t - 1] + np.cumsum(np.full(T-t, f_star - fx))

    commit_scores_1 = []
    commit_scores_2 = []
    for _ in range(t, T):
        i, j = rng.choice(K, size=2, p=x_hat)
        commit_scores_1.append(s_true[:, i])
        commit_scores_2.append(s_true[:, j])
    
    scores_1.extend(commit_scores_1)
    scores_2.extend(commit_scores_2)

    return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)


def explore_then_commit_borda_winner(
    P_users: np.ndarray,
    s_true: np.ndarray,
    T: int,
    rng: Generator,
    iters_fw: int = 300,
    exploration_frac: float = 0.25,
) -> Tuple:
    """
    Explore-then-commit algorithm that finds the arm with the highest average Borda score and commits to it.
    Exploration: For each arm i, duel it L times against a uniformly random arm j.
    Commit: Play the arm with the highest estimated total Borda score against a random opponent.
    """
    U, K, _ = P_users.shape

    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)
    scores_1 = []
    scores_2 = []

    # Phase 1: Exploration
    L_expl = int(exploration_frac * (K ** (-2/3) * T ** (2/3) * U ** (2/3) * (math.log(U*K*T)) ** (1/3)))
    
    wins_expl = np.zeros((U, K))
    counts_expl = np.zeros(K)
    
    t = 0
    for i in range(K):
        for _ in range(L_expl):
            if t >= T: break
            j = rng.choice([arm for arm in range(K) if arm != i])
            outcomes = sample_duel(P_users, i, j, rng) # shape (U,)
            
            wins_expl[:, i] += outcomes
            counts_expl[i] += 1
            
            # Calculate regret for this step
            x_hat_1 = np.zeros(K); x_hat_1[i] = 1.0
            x_hat_2 = np.ones(K)/(K-1); x_hat_2[i] = 0.0
            scores_1.append(s_true[:, i])
            scores_2.append(s_true[:, j])
            fx_val_1, _ = objective_and_grad(x_hat_1, s_true)
            fx_val_2, _ = objective_and_grad(x_hat_2, s_true)
            fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
            cumulative_regret[t] = (f_star - fx) + (cumulative_regret[t-1] if t > 0 else 0)
            t += 1
        if t >= T: break
    
    exploration_steps = t

    if t >= T:
        print("Warning: exploration filled the entire budget T for Borda Winner ETC.")
        return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)

    # Phase 2: Commit
    borda_scores_est_per_user = np.zeros((U, K))
    for i in range(K):
        if counts_expl[i] > 0:
            # This is the estimated mean Borda score for each user
            borda_scores_est_per_user[:, i] = wins_expl[:, i] / counts_expl[i]

    # Find winner based on total Borda score across all users
    total_borda_scores_est = np.sum(borda_scores_est_per_user, axis=0)
    winner_arm = np.argmax(total_borda_scores_est)
    
    x_hat_1 = np.zeros(K); x_hat_1[winner_arm] = 1.0
    
    fx_val_1, _ = objective_and_grad(x_hat_1, s_true)
    fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_1)
    
    cumulative_regret[t:] = cumulative_regret[t - 1] + np.cumsum(np.full(T-t, f_star - fx))

    for _ in range(t, T):
        scores_1.append(s_true[:, winner_arm])
        scores_2.append(s_true[:, winner_arm])

    return cumulative_regret, exploration_steps, np.array(scores_1), np.array(scores_2)