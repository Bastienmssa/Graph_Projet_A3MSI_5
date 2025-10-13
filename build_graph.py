import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

# Import des modules dédiés
from load_data import load_real_data, load_queries
from weighted_distance import weighted_distance
from radiusx_search import nodes_in_radius, process_queries

def make_features(N, d=50, rng=None):
    """
    FONCTION LEGACY - Génère un tableau (N, d) de 50 caractéristiques synthétiques.
    Utilisez load_real_data() à la place pour les données réelles.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X = np.zeros((N, d), dtype=float)

    # Démographie (5)
    age = np.clip(rng.normal(35, 10, size=N), 18, 75)
    income = np.clip(rng.lognormal(mean=10, sigma=0.5, size=N), 1e3, 3e5)
    visits_per_week = np.clip(rng.normal(10, 4, size=N), 0, 50)
    session_duration = np.clip(rng.normal(5, 2, size=N), 0.1, 30)  # minutes
    conversion_rate = np.clip(rng.beta(2, 20, size=N), 0, 1)

    X[:, 0] = age
    X[:, 1] = income
    X[:, 2] = visits_per_week
    X[:, 3] = session_duration
    X[:, 4] = conversion_rate

    # Intérêts (30) ~ Dirichlet par secteur latent
    n_interests = 30
    n_topics = 6
    topic_weights = rng.dirichlet(alpha=np.ones(n_topics), size=N)
    topic_basis = rng.dirichlet(alpha=np.ones(n_interests), size=n_topics)  # profils thématiques
    interests = topic_weights @ topic_basis
    interests += rng.normal(0, 0.02, size=interests.shape)
    interests = np.clip(interests, 0, None)
    interests = interests / (interests.sum(axis=1, keepdims=True) + 1e-12)
    X[:, 5:5+n_interests] = interests

    # Device/Browser (5) quasi one-hot
    for col in range(35, 40):
        X[:, col] = rng.uniform(0, 0.1, size=N)
    device_idx = rng.integers(35, 40, size=N)
    X[np.arange(N), device_idx] += 0.9

    # Géographie (5) quasi one-hot
    for col in range(40, 45):
        X[:, col] += rng.uniform(0, 0.1, size=N)
    geo_idx = rng.integers(40, 45, size=N)
    X[np.arange(N), geo_idx] += 0.9

    # Budget marketing (1)
    X[:, 45] = np.clip(rng.lognormal(mean=7, sigma=0.8, size=N), 50, 5e4)

    # Signaux comportementaux (4)
    X[:, 46:50] = np.clip(rng.normal(loc=[0.3, 0.5, 0.2, 0.1], scale=0.15, size=(N, 4)), 0, 1)

    return X

def build_graph(
    data_file="Data/adsSim_data_nodes.csv",
    edge_mode="knn",
    k=10,
    threshold=4.5,
    sector_count=8,
    p_in=0.12,
    p_out=0.01,
    seed=42,
    use_synthetic=False,
    N=1000,
    d=50
):
    """
    Construit un graphe G et les features X.
    
    Paramètres:
    - data_file: fichier CSV avec les données réelles
    - use_synthetic: si True, utilise make_features() au lieu des données réelles
    - edge_mode appartient à {"knn","threshold","sector","random"}
    - "knn": k plus proches voisins (sur features normalisées)
    - "threshold": arête si distance euclidienne ≤ threshold
    - "sector": communautés (stochastic block model) + features corrélées
    - "random": G(n,p) avec p approx k/N
    """
    rng = np.random.default_rng(seed)
    
    if use_synthetic:
        # Mode legacy avec données synthétiques
        X = make_features(N, d=d, rng=rng)
        node_ids = np.arange(N)
        cluster_ids = None
        N = len(X)
    else:
        # Charger les données réelles
        X, node_ids, cluster_ids = load_real_data(data_file)
        N = len(X)
        d = X.shape[1]

    # Normalisation pour des distances stables
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    meta = {}

    if edge_mode == "knn":
        nbrs = NearestNeighbors(n_neighbors=min(k+1, N), algorithm="auto").fit(Xn)
        dist, idx = nbrs.kneighbors(Xn, return_distance=True)
        # idx[:,0] = le nœud lui-même -> on part de 1
        for i in range(N):
            for j, d_ij in zip(idx[i, 1:], dist[i, 1:]):
                G.add_edge(i, j, weight=float(d_ij))

    elif edge_mode == "threshold":
        # Approx rapide : on utilise kNN large puis on filtre par seuil
        k_large = min(max(50, k*3), N-1)
        nbrs = NearestNeighbors(n_neighbors=k_large+1).fit(Xn)
        dist, idx = nbrs.kneighbors(Xn, return_distance=True)
        for i in range(N):
            for j, d_ij in zip(idx[i, 1:], dist[i, 1:]):
                if d_ij <= threshold:
                    G.add_edge(i, j, weight=float(d_ij))

    elif edge_mode == "sector":
        # Attribution sectorielle basée sur cluster_ids si disponibles
        if cluster_ids is not None:
            # Utiliser les clusters réels
            sector_of = cluster_ids.copy()
            unique_sectors = np.unique(sector_of)
            blocks = []
            for s in unique_sectors:
                blocks.append(np.where(sector_of == s)[0].tolist())
        else:
            # Attribution sectorielle aléatoire (mode legacy)
            sizes = np.full(sector_count, N // sector_count)
            sizes[: N % sector_count] += 1
            blocks = []
            start = 0
            sector_of = np.empty(N, dtype=int)
            for s, sz in enumerate(sizes):
                blocks.append(list(range(start, start+sz)))
                sector_of[start:start+sz] = s
                start += sz
        meta["sector"] = sector_of

        # Connexions intra/inter
        for s, nodes in enumerate(blocks):
            for i in nodes:
                for j in nodes:
                    if j <= i: 
                        continue
                    if rng.random() < p_in:
                        G.add_edge(i, j, weight=1.0)

        # inter-secteurs clairsemé
        all_nodes = list(range(N))
        for _ in range(int(N * k * 0.5)):  # quelques ponts
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if sector_of[i] != sector_of[j] and rng.random() < p_out:
                G.add_edge(i, j, weight=1.0)

        # Optionnel: ajuster poids par distance sur Xn
        for u, v in G.edges():
            d_uv = np.linalg.norm(Xn[u] - Xn[v])
            G[u][v]["weight"] = float(d_uv)

    elif edge_mode == "random":
        p = min(1.0, max(0.0001, k / N))  # degré moyen ~ k
        G = nx.gnp_random_graph(N, p, seed=seed)
        # poids = distance sur Xn (utile pour plus tard)
        for u, v in G.edges():
            d_uv = np.linalg.norm(Xn[u] - Xn[v])
            G[u][v]["weight"] = float(d_uv)

    else:
        raise ValueError("edge_mode must be 'knn', 'threshold', 'sector', or 'random'")

    features = {i: X[i] for i in range(N)}
    return G, X, features, meta, scaler, node_ids, cluster_ids

# ---- Distance pondérée d_Y(A, v) = sqrt( sum_k y_k * (f_Ak - f_vk)^2 ) ----
# Fonction déplacée vers weighted_distance.py

# ---- Fonctions de recherche dans un rayon ----
# Fonctions déplacées vers radiusx_search.py