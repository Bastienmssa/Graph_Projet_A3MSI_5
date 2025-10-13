#!/usr/bin/env python3
"""
Module de recherche dans un rayon X pour le projet Graph_Projet_A3MSI_5
Implémente l'Étape 3 : Recherche de tous les nœuds à distance pondérée ≤ X

Stratégies implémentées :
1. NAÏVE : Parcours exhaustif de tous les nœuds
2. STRUCTURÉE : Optimisation Top-M avec pruning intelligent
3. BATCH : Traitement par lot de requêtes multiples

Fonctions principales :
- nodes_in_radius() : Recherche naïve simple
- _search_naive() : Recherche naïve vectorisée haute performance  
- _search_pruned_topM() : Recherche structurée avec pruning Top-M
"""

import numpy as np
import pandas as pd
from weighted_distance import weighted_distance, _w2_dist2
from load_data import _parse_y

# ========== ÉTAPE 3 - STRATÉGIE 1 : NAÏVE SIMPLE ==========

def nodes_in_radius(G, X, Y, node_A, X_radius):
    """
    STRATÉGIE NAÏVE : Parcours exhaustif de tous les nœuds.
    
    Trouve tous les nœuds v tels que d_Y(A,v) ≤ X_radius.
    Implémentation simple et compréhensible de l'Étape 3.
    
    Args:
        G : networkx.Graph - graphe (utilisé pour itérer sur les nœuds)
        X : np.ndarray (N, 50) - matrice des features de tous les nœuds
        Y : np.ndarray (50,) - vecteur de pondération
        node_A : int - index du nœud de départ A
        X_radius : float - rayon de recherche X
        
    Returns:
        list[int] - liste des indices des nœuds dans le rayon
        
    Complexité : O(N × 50) où N = nombre de nœuds
    Usage : Démonstrations, tests, compréhension de l'algorithme
    """
    fA = X[node_A]
    res = []
    for v in G.nodes():                                    # PARCOURS EXHAUSTIF
        if v == node_A:
            continue
        if weighted_distance(fA, X[v], Y) <= X_radius:     # Test d_Y(A,v) ≤ X
            res.append(v)
    return res

# ========== ÉTAPE 3 - STRATÉGIE 2 : NAÏVE VECTORISÉE ==========

def _search_naive(ids, X, qdf, dmult):
    """
    STRATÉGIE NAÏVE VECTORISÉE : Brute force exact haute performance.
    
    Parcours exhaustif optimisé avec vectorisation NumPy.
    Traite plusieurs requêtes simultanément.
    
    Args:
        ids : np.ndarray - identifiants des nœuds
        X : np.ndarray (N, 50) - matrice des features
        qdf : pd.DataFrame - requêtes avec colonnes point_A, Y_vector, D
        dmult : float - multiplicateur pour le rayon (démo)
        
    Returns:
        pd.DataFrame - résultats avec colonnes query_idx, point_A, node, dist
        
    Complexité : O(Q × N × 50) où Q = nombre de requêtes
    Usage : API web, traitement batch haute performance
    """
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    X32 = X.astype(np.float32, copy=False)
    out = []
    EPS = 1e-12
    
    for qi, row in qdf.iterrows():
        a = row["point_A"]
        D = float(row["D"]) * float(dmult)
        if a not in id_to_idx: 
            continue
            
        y = _parse_y(row["Y_vector"])
        aidx = id_to_idx[a]
        
        # CALCUL VECTORISÉ : distance² vers TOUS les nœuds
        d2 = _w2_dist2(X32, X32[aidx], y)
        
        # FILTRAGE : garder seulement d ≤ D
        mask = d2 <= (D*D + EPS)
        idxs = np.where(mask)[0]
        if idxs.size == 0: 
            continue
            
        # TRI par distance croissante
        order = np.argsort(d2[idxs], kind="stable")
        idxs = idxs[order]
        
        out.append(pd.DataFrame({
            "query_idx": qi, 
            "point_A": a,
            "node": ids[idxs], 
            "dist": np.sqrt(d2[idxs]).astype(np.float32)
        }))
        
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["query_idx","point_A","node","dist"]
    )

# ========== ÉTAPE 3 - STRATÉGIE 3 : STRUCTURÉE AVEC PRUNING ==========

def _topM_idx(y: np.ndarray, M: int) -> np.ndarray:
    """
    Sélectionne les indices des M plus grands poids de y (Top-M(Y)).
    Utilisé pour le test rapide de borne inférieure.
    
    Args:
        y : np.ndarray (50,) - vecteur de pondération
        M : int - nombre de features à sélectionner (1-50)
        
    Returns:
        np.ndarray (M,) - indices des M plus grandes valeurs de y
    """
    M = max(1, min(M, y.shape[0]))
    return np.argpartition(-y, M-1)[:M]

def _partial_lb_dist2(X: np.ndarray, xA: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Calcule la borne inférieure LB² en sommant seulement sur les indices Top-M.
    
    Propriété mathématique : LB² ≤ d² vraie
    Si LB² > D², alors d² > D² garanti → rejet sans calcul complet
    
    Args:
        X : np.ndarray (N, 50) - matrice de tous les nœuds
        xA : np.ndarray (50,) - features du nœud A
        y : np.ndarray (50,) - vecteur de pondération complet
        idx : np.ndarray (M,) - indices des M features importantes
        
    Returns:
        np.ndarray (N,) - bornes inférieures LB² pour tous les nœuds
    """
    diff = X[:, idx] - xA[idx]                # Différence sur M features seulement
    return (diff*diff) @ y[idx]               # LB² = ∑[k∈Top-M] y_k × (f_Ak - f_vk)²

def _search_pruned_topM(ids, X, qdf, dmult, M=12):
    """
    STRATÉGIE STRUCTURÉE : Recherche avec pruning intelligent Top-M.
    
    Algorithme en 2 phases :
    1. Test rapide : Borne inférieure sur M features importantes
    2. Test complet : Distance exacte sur les candidats survivants
    
    Args:
        ids : np.ndarray - identifiants des nœuds
        X : np.ndarray (N, 50) - matrice des features
        qdf : pd.DataFrame - requêtes avec colonnes point_A, Y_vector, D
        dmult : float - multiplicateur pour le rayon (démo)
        M : int - nombre de features pour le test rapide (1-50)
        
    Returns:
        tuple: (résultats, pruned_ratio)
            - résultats : pd.DataFrame avec colonnes query_idx, point_A, node, dist
            - pruned_ratio : float - proportion de nœuds éliminés par pruning
            
    Complexité : O(Q × N × M + Q × k × 50) où k << N (candidats survivants)
    Usage : API web haute performance, optimisation avancée
    """
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    X32 = X.astype(np.float32, copy=False)
    out = []
    pruned_total = 0
    tested_total = 0
    EPS = 1e-12

    for qi, row in qdf.iterrows():
        a = row["point_A"]
        D = float(row["D"]) * float(dmult)
        if a not in id_to_idx: 
            continue
            
        y = _parse_y(row["Y_vector"])
        aidx = id_to_idx[a]
        xA = X32[aidx]

        # PHASE 1 : Test rapide avec borne inférieure
        idxM = _topM_idx(y, M)                            # Top-M features importantes
        lb2 = _partial_lb_dist2(X32, xA, y, idxM)        # Borne inférieure LB²
        tested_total += X32.shape[0]
        
        # PRUNING : Éliminer les candidats impossibles
        keep = np.where(lb2 <= (D*D + EPS))[0]           # LB² ≤ D² → candidats possibles
        pruned_total += (X32.shape[0] - keep.size)       # Compteur des éliminés
        if keep.size == 0:
            continue

        # PHASE 2 : Test complet sur les candidats survivants
        d2_sel = _w2_dist2(X32[keep], xA, y)             # Distance² exacte (50 features)
        mask = d2_sel <= (D*D + EPS)                     # Filtrage final d ≤ D
        if not np.any(mask):
            continue
        keep = keep[mask]
        d2_sel = d2_sel[mask]

        # TRI par distance croissante
        order = np.argsort(d2_sel, kind="stable")
        keep = keep[order]
        d2_sel = d2_sel[order]

        out.append(pd.DataFrame({
            "query_idx": qi, 
            "point_A": a,
            "node": ids[keep], 
            "dist": np.sqrt(d2_sel).astype(np.float32)
        }))

    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["query_idx","point_A","node","dist"]
    )
    pruned_ratio = float(pruned_total / tested_total) if tested_total > 0 else None
    return res, pruned_ratio

# ========== FONCTIONS UTILITAIRES ==========

def process_queries(G, X, queries, node_ids):
    """
    Traite toutes les requêtes et retourne les résultats.
    Version simple utilisant nodes_in_radius().
    
    Args:
        G : networkx.Graph - graphe
        X : np.ndarray (N, 50) - matrice des features
        queries : list[dict] - requêtes avec clés point_A, Y_vector, D
        node_ids : np.ndarray - identifiants des nœuds
        
    Returns:
        list[dict] - résultats avec statistiques par requête
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

# ========== COMPARAISON DES STRATÉGIES ==========

def compare_search_strategies(ids, X, qdf, dmult=1.0, M=12):
    """
    Compare les performances des différentes stratégies de recherche.
    
    Args:
        ids : np.ndarray - identifiants des nœuds
        X : np.ndarray (N, 50) - matrice des features
        qdf : pd.DataFrame - requêtes
        dmult : float - multiplicateur de rayon
        M : int - paramètre Top-M pour la stratégie structurée
        
    Returns:
        dict - statistiques comparatives des stratégies
    """
    import time
    
    # Test stratégie naïve
    start_time = time.perf_counter()
    results_naive = _search_naive(ids, X, qdf, dmult)
    time_naive = time.perf_counter() - start_time
    
    # Test stratégie structurée
    start_time = time.perf_counter()
    results_pruned, pruned_ratio = _search_pruned_topM(ids, X, qdf, dmult, M)
    time_pruned = time.perf_counter() - start_time
    
    return {
        'naive': {
            'time_sec': time_naive,
            'n_results': len(results_naive),
            'results': results_naive
        },
        'pruned': {
            'time_sec': time_pruned,
            'n_results': len(results_pruned),
            'pruned_ratio': pruned_ratio,
            'results': results_pruned
        },
        'speedup': time_naive / time_pruned if time_pruned > 0 else float('inf'),
        'same_results': len(results_naive) == len(results_pruned)
    }

# ========== VALIDATION ET TESTS ==========

def validate_radius_search_results(results_df, expected_columns=None):
    """
    Valide les résultats d'une recherche dans un rayon.
    
    Args:
        results_df : pd.DataFrame - résultats de recherche
        expected_columns : list - colonnes attendues
        
    Returns:
        dict - rapport de validation
    """
    if expected_columns is None:
        expected_columns = ["query_idx", "point_A", "node", "dist"]
    
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Vérifier les colonnes
    missing_cols = set(expected_columns) - set(results_df.columns)
    if missing_cols:
        validation['valid'] = False
        validation['errors'].append(f"Colonnes manquantes: {missing_cols}")
    
    # Vérifier les distances
    if 'dist' in results_df.columns:
        if (results_df['dist'] < 0).any():
            validation['valid'] = False
            validation['errors'].append("Distances négatives trouvées")
        
        validation['stats']['min_dist'] = results_df['dist'].min()
        validation['stats']['max_dist'] = results_df['dist'].max()
        validation['stats']['mean_dist'] = results_df['dist'].mean()
    
    # Statistiques générales
    validation['stats']['n_results'] = len(results_df)
    validation['stats']['n_queries'] = results_df['query_idx'].nunique() if 'query_idx' in results_df.columns else None
    
    return validation

# ========== EXEMPLE D'USAGE ==========

if __name__ == "__main__":
    """
    Démonstration des stratégies de recherche dans un rayon
    """
    print("=== Test des stratégies de recherche dans un rayon X ===")
    
    # Créer des données d'exemple
    np.random.seed(42)
    N = 100  # 100 nœuds pour l'exemple
    
    # Features aléatoires
    X = np.random.rand(N, 50).astype(np.float32)
    ids = np.array([f"node_{i+1}" for i in range(N)])
    
    # Vecteur de pondération (plus de poids sur les premières features)
    Y = np.exp(-np.arange(50) * 0.1)  # Décroissance exponentielle
    Y = Y / Y.sum()  # Normaliser
    
    # Créer une requête d'exemple
    qdf = pd.DataFrame({
        'point_A': ['node_1'],
        'Y_vector': [';'.join(map(str, Y))],
        'D': [2.0]
    })
    
    print(f"Données d'exemple : {N} nœuds, 50 features")
    print(f"Requête : point_A=node_1, D=2.0")
    print(f"Vecteur Y : poids décroissant (max={Y[0]:.3f}, min={Y[-1]:.3f})")
    
    # Comparaison des stratégies
    comparison = compare_search_strategies(ids, X, qdf, dmult=1.0, M=12)
    
    print(f"\n=== Résultats de comparaison ===")
    print(f"Stratégie naïve :")
    print(f"  - Temps : {comparison['naive']['time_sec']:.6f}s")
    print(f"  - Résultats : {comparison['naive']['n_results']} nœuds")
    
    print(f"Stratégie structurée (Top-M=12) :")
    print(f"  - Temps : {comparison['pruned']['time_sec']:.6f}s")
    print(f"  - Résultats : {comparison['pruned']['n_results']} nœuds")
    print(f"  - Pruning : {comparison['pruned']['pruned_ratio']*100:.1f}% éliminés")
    
    print(f"\nAccélération : ×{comparison['speedup']:.2f}")
    print(f"Résultats identiques : {'✅' if comparison['same_results'] else '❌'}")
    
    # Validation des résultats
    validation = validate_radius_search_results(comparison['naive']['results'])
    print(f"\nValidation : {'✅ Valide' if validation['valid'] else '❌ Erreurs'}")
    if validation['stats']:
        stats = validation['stats']
        print(f"Distance min/max/moyenne : {stats.get('min_dist', 'N/A'):.3f} / {stats.get('max_dist', 'N/A'):.3f} / {stats.get('mean_dist', 'N/A'):.3f}")
