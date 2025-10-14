# radiusx_search.py
"""
Module pour la recherche dans le rayon X.
Contient toutes les fonctions liées à l'étape 3 du projet : Recherche dans le rayon X.

Implémente différentes stratégies :
- Naïve : parcours exhaustif de tous les nœuds
- Structurée (pruned) : parcours optimisé avec borne inférieure Top-M
"""

import time
import numpy as np
import pandas as pd
from io import StringIO

# Import des modules du projet
from weighted_distance import weighted_squared_distance
from graph_build import (
    parse_y_vector, default_y_from_queries_csv, 
    infer_graph_prefix, normalize_node_id
)

def radius_search_naive(ids: np.ndarray, X: np.ndarray, qdf: pd.DataFrame, 
                       colA: str, dscale: float = 1.0):
    """
    Recherche naïve dans le rayon X : parcours exhaustif de tous les nœuds.
    
    Pour chaque query (A, Y, D), trouve tous les nœuds B tels que d_Y(A,B) ≤ D*dscale.
    
    Args:
        ids: Identifiants des nœuds (N,)
        X: Matrice des features (N, 50)
        qdf: DataFrame des queries avec colonnes point_A, Y_vector, D
        colA: Nom de la colonne contenant les points A
        dscale: Facteur d'échelle pour le rayon (défaut: 1.0)
        
    Returns:
        tuple: (résultats, temps_calcul, queries_matchées, queries_ignorées)
               résultats = liste de (query_idx, point_A_original, nœud_trouvé, distance)
    """
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    graph_prefix = infer_graph_prefix(ids)
    out_rows = []
    matched = 0
    skipped = 0
    t0 = time.perf_counter()
    
    for qi, row in qdf.iterrows():
        Araw = str(row[colA])
        A = normalize_node_id(Araw, id_to_idx, graph_prefix)
        
        if A is None:
            skipped += 1
            continue
            
        matched += 1
        D = float(row["D"])
        Deff2 = float((D * dscale) ** 2)  # Rayon² effectif
        
        # Extraction du vecteur de pondération Y
        if "Y_vector" in qdf.columns and isinstance(row.get("Y_vector", None), str) and row["Y_vector"].strip():
            y = parse_y_vector(str(row["Y_vector"]))
        else:
            y = default_y_from_queries_csv(qdf.to_csv(index=False).encode())
            
        # Calcul des distances à tous les nœuds
        idxA = id_to_idx[A]
        d2 = weighted_squared_distance(X, X[idxA], y)
        d2[idxA] = np.inf  # Exclure le nœud de départ
        
        # Sélection des nœuds dans le rayon
        sel = np.where(d2 <= Deff2)[0]
        
        if sel.size:
            d = np.sqrt(d2[sel])
            order = np.argsort(d, kind="stable")  # Tri par distance croissante
            
            for j, dist in zip(sel[order], d[order]):
                out_rows.append((qi, Araw, ids[j], float(dist)))
    
    return out_rows, time.perf_counter() - t0, matched, skipped

def radius_search_pruned(ids: np.ndarray, X: np.ndarray, qdf: pd.DataFrame, 
                        colA: str, dscale: float = 1.0, top_m: int = 12):
    """
    Recherche structurée dans le rayon X : optimisation par borne inférieure Top-M.
    
    Utilise une borne inférieure calculée sur les M features de plus grand poids
    pour éliminer rapidement les nœuds hors rayon avant le calcul complet.
    
    Args:
        ids: Identifiants des nœuds (N,)
        X: Matrice des features (N, 50)
        qdf: DataFrame des queries
        colA: Nom de la colonne contenant les points A
        dscale: Facteur d'échelle pour le rayon
        top_m: Nombre de features pour la borne inférieure (1-50)
        
    Returns:
        tuple: (résultats, temps_calcul, rejets_pruning, queries_matchées, queries_ignorées)
    """
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    graph_prefix = infer_graph_prefix(ids)
    out_rows = []
    pruned_rejects = 0
    matched = 0
    skipped = 0
    t0 = time.perf_counter()
    
    for qi, row in qdf.iterrows():
        Araw = str(row[colA])
        A = normalize_node_id(Araw, id_to_idx, graph_prefix)
        
        if A is None:
            skipped += 1
            continue
            
        matched += 1
        D = float(row["D"])
        Deff2 = float((D * dscale) ** 2)
        
        # Extraction du vecteur Y
        if "Y_vector" in qdf.columns and isinstance(row.get("Y_vector", None), str) and row["Y_vector"].strip():
            y = parse_y_vector(str(row["Y_vector"]))
        else:
            y = default_y_from_queries_csv(qdf.to_csv(index=False).encode())
            
        # Sélection des Top-M features (plus grands poids)
        m = max(1, min(int(top_m), 50))
        top_idx = np.argsort(-y)[:m]
        
        # Calcul de la borne inférieure sur Top-M features
        XA = X[id_to_idx[A]]
        diffM = X[:, top_idx] - XA[top_idx]
        lb2 = (diffM * diffM) @ y[top_idx]
        lb2[id_to_idx[A]] = np.inf  # Exclure le nœud de départ
        
        # Pré-filtrage par borne inférieure
        cand = np.where(lb2 <= Deff2)[0]
        pruned_rejects += int((lb2 > Deff2).sum())
        
        if cand.size:
            # Calcul complet pour les candidats
            d2 = weighted_squared_distance(X[cand], XA, y)
            mask = d2 <= Deff2
            cand2 = cand[mask]
            
            if cand2.size:
                d = np.sqrt(d2[mask])
                order = np.argsort(d, kind="stable")
                
                for j, dist in zip(cand2[order], d[order]):
                    out_rows.append((qi, Araw, ids[j], float(dist)))
    
    return out_rows, time.perf_counter() - t0, pruned_rejects, matched, skipped

def compare_radius_search_strategies(ids: np.ndarray, X: np.ndarray, qdf: pd.DataFrame,
                                   colA: str, dscale: float = 1.0, top_m: int = 12):
    """
    Compare les performances des stratégies naïve vs structurée.
    
    Args:
        ids: Identifiants des nœuds
        X: Matrice des features
        qdf: DataFrame des queries
        colA: Nom de la colonne point_A
        dscale: Facteur d'échelle
        top_m: Paramètre Top-M pour la stratégie structurée
        
    Returns:
        dict: Statistiques de comparaison
    """
    # Exécution naïve
    N, tN, mN, sN = radius_search_naive(ids, X, qdf, colA, dscale)
    
    # Exécution structurée
    P, tP, rej, mP, sP = radius_search_pruned(ids, X, qdf, colA, dscale, top_m)
    
    # Vérification de l'égalité des résultats
    equal = (sorted(N) == sorted(P))
    
    return {
        "naive_time_sec": tN,
        "pruned_time_sec": tP,
        "rows_naive": len(N),
        "rows_pruned": len(P),
        "equal": equal,
        "pruned_rejects": rej,
        "matched_queries": mN,
        "skipped_queries": sN,
        "speedup": tN / tP if tP > 0 else float('inf'),
        "pruning_efficiency": rej / (len(ids) * mN) if mN > 0 else 0.0
    }

def radius_search_to_csv(results: list, mode: str = "naive") -> str:
    """
    Convertit les résultats de recherche en format CSV.
    
    Args:
        results: Liste de (query_idx, point_A, node, dist)
        mode: Mode de recherche pour le nom du fichier
        
    Returns:
        str: Contenu CSV
    """
    buf = StringIO()
    buf.write("query_idx,point_A,node,dist\n")
    
    for qi, A, nid, dist in results:
        buf.write(f"{qi},{A},{nid},{dist:.6f}\n")
    
    return buf.getvalue()

def validate_radius_search():
    """
    Fonction de test pour valider l'implémentation de la recherche dans le rayon.
    
    Returns:
        bool: True si tous les tests passent
    """
    # Test avec des données synthétiques
    print("🧪 Tests de validation de la recherche dans le rayon:")
    
    # Création de données de test
    ids = np.array(['node_1', 'node_2', 'node_3'])
    X = np.zeros((3, 50))
    X[0, 0] = 0.0  # node_1 à l'origine
    X[1, 0] = 3.0  # node_2 à distance 3
    X[2, 0] = 6.0  # node_3 à distance 6
    
    # Query : chercher autour de node_1 avec rayon 4
    qdf = pd.DataFrame({
        'point_A': ['node_1'],
        'Y_vector': [';'.join(['1.0'] + ['0.0'] * 49)],  # Poids sur feature_1
        'D': [4.0]
    })
    
    # Test naïve
    results_naive, _, _, _ = radius_search_naive(ids, X, qdf, 'point_A')
    
    # Test structurée
    results_pruned, _, _, _, _ = radius_search_pruned(ids, X, qdf, 'point_A')
    
    # Vérifications
    test1 = len(results_naive) == 1  # Seul node_2 dans le rayon
    test2 = len(results_pruned) == 1
    test3 = results_naive[0][2] == 'node_2'  # Le nœud trouvé est node_2
    test4 = abs(results_naive[0][3] - 3.0) < 1e-6  # Distance = 3.0
    test5 = results_naive == results_pruned  # Résultats identiques
    
    print(f"Test 1 (nombre résultats naïve): {'✅' if test1 else '❌'}")
    print(f"Test 2 (nombre résultats pruned): {'✅' if test2 else '❌'}")
    print(f"Test 3 (nœud correct): {'✅' if test3 else '❌'}")
    print(f"Test 4 (distance correcte): {'✅' if test4 else '❌'}")
    print(f"Test 5 (égalité stratégies): {'✅' if test5 else '❌'}")
    
    return test1 and test2 and test3 and test4 and test5

if __name__ == "__main__":
    # Tests automatiques
    all_tests_pass = validate_radius_search()
    print(f"\n{'🎉 Tous les tests passent!' if all_tests_pass else '❌ Certains tests échouent'}")
