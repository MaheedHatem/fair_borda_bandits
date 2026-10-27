import numpy as np
from numpy.random import Generator
from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
import scipy.optimize as opt
import os
from typing import Optional
import os
from dataclasses import dataclass
from typing import Callable, Optional
from scipy.linalg import qr
 
def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def bit_representation(x: int, d: int) -> np.ndarray:
    """
    Computes the {-1, 1}^d binary representation of an integer x.
    """
    if not 0 <= x < 2**d:
        raise ValueError(f"Input x must be in [0, 2^d - 1]. Got x={x}, d={d}")
    
    # Get binary string, 0-padded to length d
    bin_str = format(x, f'0{d}b')
    
    # Map '0' to -1 and '1' to 1
    return np.array([1 if bit == '1' else -1 for bit in bin_str], dtype=float)
 
 
def linear_link(x: np.ndarray) -> np.ndarray:
    return 0.5 + x

def generate_random_glm_instance(
    K: int,
    U: int,
    d: int,
    rng: Generator,
    link: Callable = logistic,
    theta_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]: 
    phi = np.zeros((K, K, d))
    for i in range(K):
        phi[i, i] = np.zeros(d)
        for j in range(i + 1, K):
            v = rng.standard_normal(d)
            v /= (np.linalg.norm(v) + 1e-12)  # unit vector
            phi[i, j] =  v
            phi[j, i] = -v
 
    # Generate U preference matrices, one for each user
    P_users = np.zeros((U, K, K))
    for u in range(U):
        theta_star_u = rng.standard_normal(d) * theta_scale
        for i in range(K):
            for j in range(K):
                if i == j:
                    P_users[u, i, j] = 0.5
                else:
                    P_users[u, i, j] = link(phi[i, j] @ theta_star_u)
    return P_users, phi

def generate_conflicting_glm_instance(
    K: int,
    U: int,
    d: int,
    rng: Generator,
    link: Callable = logistic,
    theta_scale: float = 1.0,
    noise_scale: float = 0.1,
    num_groups: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a GLM instance with structured user conflict.

    Users are partitioned into `num_groups` groups. Each group has an
    "archetypal" preference vector theta. Each user's theta is a noisy
    version of their group's archetype. With num_groups=2, the archetypes
    are opposites, creating two factions of users with conflicting preferences.

    Args:
        K: Number of arms.
        U: Number of users.
        d: Dimension of context vectors.
        rng: Random generator used for phi, archetypes, group assignment and noise.
        link: Link function for the GLM.
        theta_scale: Scale of the archetypal theta vectors.
        noise_scale: Scale of the noise added to each user's theta.
        num_groups: Number of user preference groups to create.

    Returns:
        A tuple containing the (U, K, K) preference matrices and the (K, K, d) feature vectors.
    """
    # 1. Generate feature vectors phi, same as in the random instance
    phi = np.zeros((K, K, d))
    for i in range(K):
        phi[i, i] = np.zeros(d)
        for j in range(i + 1, K):
            v = rng.standard_normal(d)
            v /= (np.linalg.norm(v) + 1e-12)  # unit vector
            phi[i, j] =  v
            phi[j, i] = -v

    # 2. Generate `num_groups` archetypal theta vectors
    archetype_thetas = [rng.standard_normal(d) * theta_scale for _ in range(num_groups)]
    if num_groups == 2:
        # For 2 groups, create two diametrically opposed preference archetypes
        archetype_thetas[1] = -archetype_thetas[0]

    # 3. Assign users to groups and generate their specific thetas
    P_users = np.zeros((U, K, K))
    user_groups = rng.integers(0, num_groups, size=U)
    for u in range(U):
        base_theta = archetype_thetas[user_groups[u]]
        noise = rng.standard_normal(d) * noise_scale
        theta_star_u = base_theta + noise
        
        # 4. Generate the preference matrix for user u based on their theta
        P_users[u, :, :] = link(np.einsum('ijd,d->ij', phi, theta_star_u))
        np.fill_diagonal(P_users[u, :, :], 0.5)

    return P_users, phi

def generate_user_prefs(U: int, K: int, rng: Generator) -> np.ndarray:
    """
    Generate U user-specific preference matrices over K arms.
    """
    pref_matrices = np.zeros((U,K, K))
    for u in range(U):
        for i in range(K):
            for j in range(i,K):
                if i == j:
                    pref_matrices[u, i, j] = 0.5
                else:
                    pref_matrices[u, i, j] = rng.random()
                pref_matrices[u, j, i] = 1 - pref_matrices[u, i, j]

    for u in range(U):
        assert(np.allclose(pref_matrices[u,:,:] + pref_matrices[u,:,:].T, np.ones((K,K))))
        assert(np.allclose(np.diag(pref_matrices[u,:,:]), 0.5 * np.ones(K)))

    return pref_matrices

def generate_hard_instances(
    U: int,
    K: int,
    epsilon: float,
    m_target: Optional[int] = None
) -> np.ndarray:
    """
    Generates a set of hard preference matrix instances based on Saha et al. (2021).

    This creates a multi-user version of a hard instance family. The arms are
    partitioned into a "good" set K_g and a "bad" set. In the reference instance,
    all good arms are indistinguishable. In the alternative instances, one good
    arm `m_target` is made slightly better for a subset of users.

    Args:
        U (int): Total number of users. Should be an even number.
        K (int): Total number of arms. Should be an odd number for a balanced partition.
        epsilon (float): A small parameter in (0, 0.05) to control preference strength.
        m_target (Optional[int]): The target arm to make optimal.
            - If None, generates the reference instance (I^0) where all good arms are tied.
            - If an integer, it must be an index in the "good" set of arms.
              This generates the alternative instance I^(m_target).

    Returns:
        np.ndarray: A (U, K, K) array of preference matrices.
    """
    if not (0 < epsilon < 0.05):
        raise ValueError("epsilon must be in (0, 0.05)")

    Kg_size = (K + 1) // 2
    if K % 2 == 0:
        print(f"Warning: K={K} is even. The partition sizes will be {Kg_size} and {K-Kg_size}.")

    if m_target is not None and not (0 <= m_target < Kg_size):
        raise ValueError(f"m_target must be an arm in [K_g], i.e., between 0 and {Kg_size - 1}")

    U_pert_size = U // 2
    if U % 2 != 0:
        print(f"Warning: U={U} is odd. Perturbed user group size will be {U_pert_size}.")

    # --- Create the reference preference matrix (P_ref) for a single user ---
    P_ref = np.zeros((K, K))
    k_g_indices = np.arange(Kg_size)
    k_other_indices = np.arange(Kg_size, K)

    # Block 1: i, j in K_g (good vs good)
    P_ref[np.ix_(k_g_indices, k_g_indices)] = 0.5
    # Block 2: i, j not in K_g (bad vs bad)
    P_ref[np.ix_(k_other_indices, k_other_indices)] = 0.5
    # Block 3: i in K_g, j not in K_g (good vs bad)
    P_ref[np.ix_(k_g_indices, k_other_indices)] = 0.9 - epsilon
    # Block 4: i not in K_g, j in K_g (bad vs good)
    P_ref[np.ix_(k_other_indices, k_g_indices)] = 0.1 + epsilon

    # --- Generate instances for all U users ---
    if m_target is None:
        # Reference instance I^0: all users have the same P_ref
        pref_matrices = np.tile(P_ref, (U, 1, 1))
    else:
        # Alternative instance I^(m_target):
        P_pert = P_ref.copy()
        P_pert[m_target, k_other_indices] = 0.9
        P_pert[k_other_indices, m_target] = 0.1
        pref_matrices = np.zeros((U, K, K))
        if U_pert_size > 0:
            pref_matrices[np.arange(U_pert_size), :, :] = P_pert
        if U_pert_size < U:
            pref_matrices[np.arange(U_pert_size, U), :, :] = P_ref

    for u in range(U):
        p_u = pref_matrices[u]
        assert np.allclose(p_u + p_u.T, np.ones((K, K)))
        assert np.allclose(np.diag(p_u), 0.5)

    return pref_matrices

def generate_hard_glm_instances(
    U: int,
    K: int,
    d: int,
    Delta: float,
    theta_base: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a hard GLM instance based on Wu et al. (2023), adapted for multiple users.

    Args:
        U (int): Total number of users. Should be an even number.
        K (int): Total number of arms. Must be >= 2^d.
        d (int): Base dimension of the feature vectors.
        Delta (float): Magnitude for the components of the base theta vector.
        theta_base (np.ndarray): The base d-dimensional parameter vector in {-Delta, +Delta}^d.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - P_users (np.ndarray): A (U, K, K) array of preference matrices.
        - phi_aug (np.ndarray): A (K, K, d+1) array of augmented and normalized feature vectors.
    """
    k1_limit = 2**d
    if K < 2**d:
        raise ValueError(f"K must be >= 2^d for hard GLM instance. Got K={K}, d={d}.")
    if U % 2 != 0:
        print(f"Warning: U={U} is odd for hard GLM instance. Perturbed group size will be {U//2}.")
    
    d_aug = d + 1
    
    # Precompute bit representations for arms in the first partition
    bit_vectors = {i: bit_representation(i, d) for i in range(k1_limit)}

    # Create augmented feature vectors (phi_ij, c_ij)
    phi_aug = np.zeros((K, K, d_aug))
    for i in range(K):
        for j in range(i + 1, K):
            i_in_k1 = i < k1_limit
            j_in_k1 = j < k1_limit

            phi_ij = np.zeros(d)
            c_ij = 0

            if i_in_k1 and not j_in_k1:
                phi_ij = bit_vectors[i]
                c_ij = 1
            elif not i_in_k1 and j_in_k1:
                phi_ij = -bit_vectors[j]
                c_ij = -1
            
            phi_aug[i, j, :d] = phi_ij
            phi_aug[i, j, d] = c_ij
            phi_aug[j, i, :] = -phi_aug[i, j, :]

    # Normalize phi_aug so that ||phi_aug_ij|| <= 1
    norm_factor = np.sqrt(d_aug)
    phi_aug /= norm_factor

    # Create augmented theta for perturbed users and scale it
    theta_aug_pert = np.zeros(d_aug)
    theta_aug_pert[:d] = theta_base
    theta_aug_pert[d] = 0.25
    theta_aug_pert *= norm_factor

    # Create preference matrices for all users
    P_users = np.zeros((U, K, K))
    U_pert_start_idx = U // 2

    # Unperturbed users (theta_u = 0) -> P_ij = 1/2
    P_users[:U_pert_start_idx, :, :] = 0.5

    # Perturbed users: P_ij = 1/2 + <phi_aug_ij, theta_aug_pert>
    if U_pert_start_idx < U:
        inner_products = np.einsum('ijd,d->ij', phi_aug, theta_aug_pert)
        P_users[U_pert_start_idx:, :, :] = 0.5 + inner_products
    
    return P_users, phi_aug

def get_borda_scores(pref_matrix: np.ndarray) -> np.ndarray:
    """Computes Borda scores for a single preference matrix."""
    # Borda score for arm i is sum_{j!=i} P(i,j)
    # np.sum(pref_matrix, axis=1) is sum over j of P(i,j).
    # We subtract diagonal element which is 0.5
    return np.sum(pref_matrix, axis=1) - 0.5

def borda_score_from_preferences(P: np.ndarray) -> np.ndarray:
    """
    Given preference matrices P, compute the normalized Borda score for each arm and user.
    Score for arm i for user u is Borda(i,u) / max_k Borda(k,u).
    Borda(i,u) = sum_{j!=i} P[u, i, j].
    """
    U, K, _ = P.shape
    norm_borda_scores = np.zeros((U, K))
    for u in range(U):
        user_borda_scores = get_borda_scores(P[u])
        max_borda_score = np.max(user_borda_scores)
        norm_borda_scores[u, :] = user_borda_scores / max(max_borda_score, (K-1)/2)
    return norm_borda_scores

def objective_and_grad(x: np.ndarray, scores: np.ndarray, utilitarian: bool = False) -> Tuple[float, np.ndarray]:
    """
    Compute f(x) = sum_u log(s_u^T x)  (concave), and gradient of g(x) = -f(x).
    We return f(x) and grad_g(x), where grad_g(x) = -∑_u b_u / (b_u^T x).
    If utilitarian is True, compute f(x) = mean_u s_u^T x and grad_g(x) = -mean_u s_u.
    """
    if utilitarian:
        vals = scores.dot(x)
        f = np.mean(vals)
        grad_g = -np.mean(scores, axis=0)
        return f, grad_g

    # Numeric stability: ensure positivity
    eps = 1e-12
    vals = scores.dot(x)
    vals = np.clip(vals, eps, None)
    f = np.sum(np.log(vals))
    grad_g = np.zeros_like(x)
    for u, v in enumerate(vals):
        grad_g -= scores[u] / v
    return f, grad_g

def frank_wolfe_goptimal(phi: np.ndarray, n_iter: int = 500, eps0: float = 0.05) -> np.ndarray:
    """
    Approximate G-optimal (equivalently D-optimal) design over the pairs i < j.

    phi must already be projected onto its span (Sec. 1.A), so that the
    stacked pair features have full column rank d.

    Returns
    -------
    pi : ndarray of shape (K, K): the design (sums to 1), supported on i < j,
         with at most d + n_iter nonzero entries.
    """
    K, _, d = phi.shape
    iu, ju = np.triu_indices(K, k=1)
    P = phi[iu, ju]                                   # (n_pairs, d)

    # Sparse full-rank initialization: d pairs with linearly independent features.
    _, _, piv = qr(P.T, pivoting=True)
    w = np.zeros(len(iu))
    w[piv[:d]] = 1.0 / d

    for _ in range(n_iter):
        V = (P * w[:, None]).T @ P                    # A(w)
        norms_sq = np.einsum("nk,kl,nl->n", P, np.linalg.inv(V), P)
        k = int(np.argmax(norms_sq))
        g = norms_sq[k]
        gamma = (g / d - 1.0) / (g - 1.0)             # exact line search, in [0, 1) since g >= d
        w *= (1.0 - gamma)
        w[k] += gamma

    V = (P * w[:, None]).T @ P
    g_final = np.einsum("nk,kl,nl->n", P, np.linalg.inv(V), P).max()
    assert w.min() >= 0 and abs(w.sum() - 1) < 1e-9
    assert g_final / d < 1 + eps0, f"g/d = {g_final/d:.4f} >= {1+eps0}"
    print(f"G-design: g/d = {g_final/d:.4f}, support = {np.count_nonzero(w)}")

    pi = np.zeros((K, K))
    pi[iu, ju] = w
    return pi

def frank_wolfe_simplex(scores: np.ndarray, K: int, iters: int = 500, tol: float = 2e-6, utilitarian: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Frank-Wolfe to minimize g(x) = -∑_u log(b_u^T x) over the probability simplex Δ_K.
    (Equivalently, maximize f(x) = ∑_u log(b_u^T x).)

    Returns:
        x: optimized stochastic policy in the simplex
        f_hist: history of the concave objective f(x) per iteration
    """
    x = np.ones(K) / K  # start uniform
    f_hist = []
    f_val, grad_g = objective_and_grad(x, scores, utilitarian=utilitarian)
    for t in range(iters):
        f_hist.append(f_val)

        # Linear minimization oracle over simplex: pick vertex minimizing <s, grad_g>
        k = np.abs(grad_g - np.min(grad_g)) <= 1e-12
        s = np.zeros(K)
        s[k] = 1.0/np.sum(k)
        # print("Grad", grad_g)
        # Step size (classic diminishing step sizes)
        gamma = 2.0 / (t + 2.0)

        x_new = (1 - gamma) * x + gamma * s
        # print("policy", x_new)

        # Convergence check (duality gap proxy: <x - s, grad_g> )
        # gap = float((x - s) @ grad_g)
        x = x_new
        f_val_new, grad_g = objective_and_grad(x, scores, utilitarian=utilitarian) 
        if abs(f_val_new - f_val) < tol:
            break
        f_val = f_val_new
    return x, f_hist

def sample_duel(P: np.ndarray, i: int, j: int, rng: Generator) -> np.ndarray:
    """
    Sample the outcome of a duel between i and j from preference matrix P.
    Returns 1 if i wins, 0 if j wins.
    """
    outcome = rng.random((len(P),)) < P[:, i, j]
    return outcome


def load_sushi_data(n_clusters: int, glm_features: bool = False, link: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads the Sushi dataset, clusters users, and creates preference matrices.
    Optionally, it can also extract and return GLM feature vectors.
    
    The preference P(i, j) for a cluster is the fraction of users in that cluster who prefer item i over j.
    The number of "users" returned for the bandit algorithms is `n_clusters`.
    Clustering is performed using Agglomerative Clustering with Spearman distance.
    If `glm_features` is True, it fits a GLM to the empirical cluster preferences to generate
    a set of preference matrices that are consistent with the GLM assumption.

    Args:
        n_clusters: The number of user clusters to create. This will be the number of "users" (U) for the algorithms.
        glm_features: If True, extracts and returns GLM feature vectors `phi`.
        link: The link function for the GLM, required if `glm_features` is True.

    Returns:
        pref_matrices: A (n_clusters, K, K) numpy array of preference matrices.
        rank_vectors_k: A (num_valid_users, K) numpy array of the original user rankings.
        phi: If `glm_features` is True, a (K, K, d) array of GLM features. Otherwise, None.
    """

    base_path = "data/sushi3-2016"
    filepath = os.path.join(base_path, "sushi3a.5000.10.order")

    K = 10
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sushi dataset file '{filepath}' not found. Please download it to 'data/sushi3-2016/'.")


    # Read rankings
    all_rankings = []
    with open(filepath, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 3:
                continue
            all_rankings.append(parts[2:])

    num_users = len(all_rankings)
    rank_vectors = np.full((num_users, 10), -1, dtype=int)
    for i, ranking in enumerate(all_rankings):
        for rank, item_id in enumerate(ranking):
            rank_vectors[i, item_id] = rank

    valid_users_mask = np.all(rank_vectors[:, :K] != -1, axis=1)
    rank_vectors_k = rank_vectors[valid_users_mask, :K]

    rho_matrix, _ = spearmanr(rank_vectors_k, axis=1, nan_policy='raise')
    dist_matrix = (1.0 - rho_matrix) / 2.0
    np.fill_diagonal(dist_matrix, 0)
    # Use Agglomerative Clustering with the precomputed distance matrix
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    cluster_labels = agg_clustering.fit_predict(dist_matrix)

    pref_matrices = np.zeros((n_clusters, K, K))
    condorcet_winners = np.full(n_clusters, -1, dtype=int)
    for c in range(n_clusters):
        user_indices_in_cluster = np.where(cluster_labels == c)[0]
        if len(user_indices_in_cluster) == 0:
            raise ValueError(f"Cluster {c} has no users.")

        num_users_in_cluster = len(user_indices_in_cluster)
        cluster_rank_vectors = rank_vectors_k[user_indices_in_cluster]

        pref_matrix_c = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1, K):
                i_is_preferred = np.sum(cluster_rank_vectors[:, i] < cluster_rank_vectors[:, j])
                i_score = np.sum(K - cluster_rank_vectors[:, i])/num_users_in_cluster
                j_score = np.sum(K - cluster_rank_vectors[:, j])/num_users_in_cluster
                #pref_matrix_c[i, j] = 0.5+0.5*(i_score - j_score)/K
                pref_matrix_c[i, j] = i_is_preferred/num_users_in_cluster
                pref_matrix_c[j, i] = 1 - pref_matrix_c[i, j]

        np.fill_diagonal(pref_matrix_c, 0.5)
        assert(np.allclose(pref_matrix_c[:,:] + pref_matrix_c[:,:].T, np.ones((K,K))))
        pref_matrices[c, :, :] = pref_matrix_c

    phi = None
    if glm_features:
        if link is None:
            raise ValueError("A link function must be provided for theta fitting when glm_features is True.")

        item_data_path = os.path.join(base_path, 'sushi3.idata')
        if not os.path.exists(item_data_path):
            raise FileNotFoundError(f"Sushi item data '{item_data_path}' not found.")

        # Columns 2-8 (0-indexed) are: style, major_group, minor_group, oiliness, freq_eat, price, freq_sold.
        # We use `usecols` to skip the non-numeric name column (col 1).
        raw_features = np.loadtxt(item_data_path, delimiter='\t', usecols=range(2, 9))

        # One-hot encode minor group (column 2 of raw_features, which is col 4 of original file)
        minor_groups = raw_features[:, 2].astype(int)
        one_hot_minor = np.zeros((raw_features.shape[0], 12))
        one_hot_minor[np.arange(raw_features.shape[0]), minor_groups] = 1

        # Normalize numerical features
        # oiliness (col 3 of raw, range 0-4)
        oiliness = raw_features[:, 3, np.newaxis] / 4.0
        # freq eat (col 4 of raw, range 0-3)
        freq_eat = raw_features[:, 4, np.newaxis] / 3.0

        # Assemble feature matrix for all 100 items
        all_item_features = np.hstack([
            raw_features[:, 0, np.newaxis],      # style
            raw_features[:, 1, np.newaxis],      # major group
            one_hot_minor,                       # minor group (one-hot)
            oiliness,
            freq_eat,
            raw_features[:, 5, np.newaxis],      # price (already normalized)
            raw_features[:, 6, np.newaxis],      # freq sold (already normalized)
        ])

        # Map from sushi set A (K=10) to set B (K=100)
        map_A_to_B = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 8, 8: 26, 9: 29}
        sushi_A_indices_in_B = [map_A_to_B[i] for i in range(K)]

        item_features_A = all_item_features[sushi_A_indices_in_B]
        d = item_features_A.shape[1]

        # Create and normalize phi vectors: phi_ij = feat_i - feat_j
        phi = np.zeros((K, K, d))
        for i in range(K):
            for j in range(i + 1, K):
                v = item_features_A[i] - item_features_A[j]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-9: # Avoid division by zero
                    v = v / norm_v
                phi[i, j] = v
                phi[j, i] = -v
        # Project phi onto the span of the pairwise features (rank <= K-1 < number of raw features).
        Phi_flat = phi.reshape(-1, d)
        _, s, Vt = np.linalg.svd(Phi_flat, full_matrices=False)
        r = int(np.sum(s > 1e-10 * s[0]))
        B = Vt[:r].T                       # (d, r) orthonormal basis of the span
        phi = phi @ B                      # (K, K, r): norms and antisymmetry are preserved
        print(f"Sushi GLM: raw feature dim = {d}, effective dim = {r}")
        d = r
        # Fit a theta for each cluster to the empirical preferences, then regenerate
        # the preference matrix. This "projects" the empirical data onto the GLM space.
        print("Fitting GLM parameters to Sushi data clusters...")
        glm_pref_matrices = np.zeros_like(pref_matrices)
        for c in range(n_clusters):
            empirical_P = pref_matrices[c]

            def loss_fn(theta):
                pred_P = link(np.einsum('ijd,d->ij', phi, theta))
                error = pred_P - empirical_P
                np.fill_diagonal(error, 0)
                return np.sum(error**2)

            res = opt.minimize(loss_fn, np.zeros(d), method='L-BFGS-B', options={'ftol': 1e-12})
            theta_c = res.x
            
            # Regenerate preference matrix for the cluster using the fitted theta
            glm_pref_matrices[c] = link(np.einsum('ijd,d->ij', phi, theta_c))
            np.fill_diagonal(glm_pref_matrices[c], 0.5)

        # Overwrite empirical matrices with the new GLM-consistent ones
        pref_matrices = glm_pref_matrices
        print("Done fitting.")

    return pref_matrices, rank_vectors_k, phi
