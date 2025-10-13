import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from math import sqrt
import os

def load_real_data(data_file="Data/adsSim_data_nodes.csv"):
    """
    Charge les données réelles depuis le fichier CSV.
    Retourne X (features), node_ids, et cluster_ids si disponibles.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Fichier de données non trouvé: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Extraire les features (colonnes feature_1 à feature_50)
    feature_cols = [f"feature_{i}" for i in range(1, 51)]
    X = df[feature_cols].values
    
    # Node IDs
    node_ids = df['node_id'].values if 'node_id' in df.columns else np.arange(len(df))
    
    # Cluster IDs si disponibles
    cluster_ids = df['cluster_id'].values if 'cluster_id' in df.columns else None
    
    return X, node_ids, cluster_ids

def load_queries(queries_file="Data/queries_structured.csv"):
    """
    Charge les requêtes structurées avec vecteurs Y et distances D.
    """
    if not os.path.exists(queries_file):
        raise FileNotFoundError(f"Fichier de requêtes non trouvé: {queries_file}")
    
    df = pd.read_csv(queries_file)
    
    # Nettoyer les noms de colonnes (enlever les espaces)
    df.columns = df.columns.str.strip()
    
    queries = []
    for _, row in df.iterrows():
        point_A = row['point_A'].strip()
        Y_vector_str = row['Y_vector'].strip()
        D = float(row['D'])
        
        # Parser le vecteur Y (format: "0.1;0.2;0.3;...")
        Y = np.array([float(x) for x in Y_vector_str.split(';')])
        
        queries.append({
            'point_A': point_A,
            'Y_vector': Y,
            'D': D
        })
    
    return queries

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
    - edge_mode ∈ {"knn","threshold","sector","random"}
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
def weighted_distance(fA, fv, Y):
    diff = fA - fv
    return sqrt(np.sum(Y * (diff ** 2)))

def nodes_in_radius(G, X, Y, node_A, X_radius):
    """
    Retourne la liste des nœuds v tels que dY(A,v) ≤ X_radius.
    Ne dépend pas de la structure du graphe pour le filtrage, 
    mais tu peux ensuite restreindre au composant de A si tu veux.
    """
    fA = X[node_A]
    res = []
    for v in G.nodes():
        if v == node_A:
            continue
        if weighted_distance(fA, X[v], Y) <= X_radius:
            res.append(v)
    return res

def process_queries(G, X, queries, node_ids):
    """
    Traite toutes les requêtes et retourne les résultats.
    """
    results = []
    
    for query in queries:
        point_A_name = query['point_A']
        Y = query['Y_vector']
        D = query['D']
        
        # Trouver l'index du nœud A
        if point_A_name.startswith('node_'):
            # Format: node_1, node_2, etc.
            node_num = int(point_A_name.split('_')[1])
            A_idx = node_num - 1  # Les nodes sont 1-indexés dans le CSV
        elif point_A_name.startswith('ads_'):
            # Format: ads_1, ads_2, etc. - mapper vers les indices de nœuds
            ads_num = int(point_A_name.split('_')[1])
            A_idx = ads_num - 1  # Supposer mapping direct pour l'instant
        else:
            print(f"Format de point_A non reconnu: {point_A_name}")
            continue
            
        if A_idx >= len(X) or A_idx < 0:
            print(f"Index de nœud invalide: {A_idx} pour {point_A_name}")
            continue
            
        # Trouver les nœuds dans le rayon
        nodes_in_range = nodes_in_radius(G, X, Y, A_idx, D)
        
        results.append({
            'point_A': point_A_name,
            'A_idx': A_idx,
            'D': D,
            'nodes_found': len(nodes_in_range),
            'nodes_list': nodes_in_range[:10]  # Limiter l'affichage
        })
    
    return results

# ------------- Démo avec données réelles -------------
if __name__ == "__main__":
    print("=== Construction du graphe avec données réelles ===")
    
    # Construire le graphe avec les données réelles
    G, X, features, meta, scaler, node_ids, cluster_ids = build_graph(
        data_file="Data/adsSim_data_nodes.csv",
        edge_mode="knn",   # "knn" | "threshold" | "sector" | "random"
        k=12,
        threshold=4.2,
        sector_count=10,
        p_in=0.10,
        p_out=0.02,
        seed=7,
        use_synthetic=False  # Utiliser les vraies données
    )

    print(f"Nœuds: {G.number_of_nodes()}, Arêtes: {G.number_of_edges()}")
    print(f"Dimensions des features: {X.shape}")
    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        print(f"Clusters trouvés: {unique_clusters}")
    
    comp_sizes = [len(c) for c in nx.connected_components(G)]
    print(f"Composantes connexes: {len(comp_sizes)} (taille max={max(comp_sizes)})")

    print("\n=== Traitement des requêtes ===")
    
    # Charger et traiter les requêtes
    try:
        queries = load_queries("Data/queries_structured.csv")
        print(f"Nombre de requêtes chargées: {len(queries)}")
        
        # Traiter quelques requêtes d'exemple
        sample_queries = queries[:5]  # Prendre les 5 premières
        results = process_queries(G, X, sample_queries, node_ids)
        
        for result in results:
            print(f"Requête {result['point_A']}: {result['nodes_found']} nœuds trouvés dans le rayon {result['D']}")
            
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Utilisation d'un exemple synthétique...")
        
        # Exemple avec vecteur Y synthétique
        rng = np.random.default_rng(7)
        Y = rng.uniform(0.0, 1.0, size=50)
        Y = Y / (Y.sum() + 1e-12)

        A = 0          # nœud de départ
        X_radius = 2.0 # rayon d_Y
        res = nodes_in_radius(G, X, Y, A, X_radius)
        print(f"Nœuds dans le rayon {X_radius} autour de A={A}: {len(res)}")
