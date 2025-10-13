#!/usr/bin/env python3
"""
Fichier principal de démonstration du projet Graph_Projet_A3MSI_5
Simulation de recherche publicitaire sur graphe pondéré

Ce script démontre :
- Construction du graphe avec données réelles
- Traitement des requêtes avec distance pondérée
- Exemple de recherche dans un rayon donné
"""

import numpy as np
import networkx as nx
from build_graph import build_graph
from load_data import load_queries
from weighted_distance import weighted_distance
from radiusx_search import process_queries, nodes_in_radius

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
