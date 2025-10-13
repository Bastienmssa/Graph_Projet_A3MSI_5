#!/usr/bin/env python3
"""
Module de calcul de distance pondérée pour le projet Graph_Projet_A3MSI_5
Implémente la formule mathématique : d_Y(u,v) = √(∑[k=1 to 50] y_k × (f_uk - f_vk)²)

Fonctions disponibles :
- weighted_distance() : Version simple pour une paire de nœuds
- _w2_dist2() : Version vectorisée optimisée pour tous les nœuds
"""

import numpy as np
from math import sqrt

# ========== IMPLÉMENTATION DE L'ÉTAPE 2 ==========

def weighted_distance(fA, fv, Y):
    """
    Calcule la distance pondérée entre deux nœuds selon la formule de l'Étape 2.
    
    Formule : d_Y(u,v) = √(∑[k=1 to 50] y_k × (f_uk - f_vk)²)
    
    Args:
        fA : np.ndarray (50,) - vecteur de features du nœud A
        fv : np.ndarray (50,) - vecteur de features du nœud v  
        Y : np.ndarray (50,) - vecteur de pondération
        
    Returns:
        float: Distance pondérée d_Y(A,v)
        
    Usage:
        Utilisée dans build_graph.py pour les recherches locales et démonstrations.
        Idéale pour comprendre l'algorithme et les calculs ponctuels.
    """
    diff = fA - fv                           # (f_Ak - f_vk) pour k=1..50
    return sqrt(np.sum(Y * (diff ** 2)))     # √(∑ y_k × (f_Ak - f_vk)²)

def _w2_dist2(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcule d²(A,B) pondérée pour TOUS les nœuds B simultanément (version optimisée).
    
    Formule : d²_Y(A,B) = ∑[k=1 to 50] y_k × (f_Ak - f_Bk)²
    Note : Retourne d² (sans racine carrée) pour éviter les calculs inutiles
    
    Args:
        X : np.ndarray (N, 50) - matrice de TOUS les nœuds
        xA : np.ndarray (50,) - vecteur de features du nœud A
        y : np.ndarray (50,) - vecteur de pondération
        
    Returns:
        np.ndarray (N,) - distances² pondérées de A vers tous les nœuds
        
    Usage:
        Utilisée dans apiameliore.py pour les recherches haute performance.
        Vectorisation NumPy : calcule 1000+ distances en une seule opération.
        
    Performance:
        ~1000x plus rapide que des appels répétés à weighted_distance()
    """
    diff = X - xA                            # (X - xA) pour tous les nœuds
    return (diff*diff) @ y                   # (diff²) @ y = ∑ y_k × (f_Ak - f_Bk)²

# ========== FONCTIONS UTILITAIRES ==========

def weighted_distance_from_squared(d2_value):
    """
    Convertit une distance² en distance (ajoute la racine carrée).
    
    Args:
        d2_value : float ou np.ndarray - valeur(s) de distance²
        
    Returns:
        float ou np.ndarray - distance(s) pondérée(s)
    """
    return np.sqrt(d2_value)

def batch_weighted_distances(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Version complète : calcule d_Y(A,B) pour tous les nœuds B (avec racine carrée).
    
    Combine _w2_dist2() + sqrt() pour obtenir les distances finales.
    
    Args:
        X : np.ndarray (N, 50) - matrice de tous les nœuds
        xA : np.ndarray (50,) - vecteur de features du nœud A
        y : np.ndarray (50,) - vecteur de pondération
        
    Returns:
        np.ndarray (N,) - distances pondérées de A vers tous les nœuds
    """
    d2 = _w2_dist2(X, xA, y)
    return np.sqrt(d2)

# ========== VALIDATION ==========

def validate_inputs(fA, fv_or_X, Y):
    """
    Valide les entrées pour les fonctions de distance pondérée.
    
    Args:
        fA : np.ndarray - features du nœud A
        fv_or_X : np.ndarray - features d'un nœud ou matrice de tous les nœuds
        Y : np.ndarray - vecteur de pondération
        
    Raises:
        ValueError: Si les dimensions ne correspondent pas
    """
    if fA.shape[0] != 50:
        raise ValueError(f"fA doit avoir 50 features (trouvé {fA.shape[0]})")
    
    if Y.shape[0] != 50:
        raise ValueError(f"Y doit avoir 50 coefficients (trouvé {Y.shape[0]})")
    
    if len(fv_or_X.shape) == 1:  # Un seul nœud
        if fv_or_X.shape[0] != 50:
            raise ValueError(f"fv doit avoir 50 features (trouvé {fv_or_X.shape[0]})")
    else:  # Matrice de nœuds
        if fv_or_X.shape[1] != 50:
            raise ValueError(f"X doit avoir 50 features par nœud (trouvé {fv_or_X.shape[1]})")

# ========== EXEMPLES D'USAGE ==========

if __name__ == "__main__":
    """
    Démonstration des fonctions de distance pondérée
    """
    print("=== Test des fonctions de distance pondérée ===")
    
    # Créer des données d'exemple
    np.random.seed(42)
    N = 5  # 5 nœuds pour l'exemple
    
    # Features aléatoires (5 nœuds × 50 features)
    X = np.random.rand(N, 50).astype(np.float32)
    
    # Vecteur de pondération aléatoire
    Y = np.random.rand(50).astype(np.float32)
    Y = Y / Y.sum()  # Normaliser
    
    # Nœud A (premier nœud)
    xA = X[0]
    
    print(f"Matrice X : {X.shape}")
    print(f"Vecteur Y : {Y.shape}")
    print(f"Nœud A : {xA.shape}")
    
    # Test 1: Distance simple (A vers nœud 1)
    d_simple = weighted_distance(xA, X[1], Y)
    print(f"\nDistance A->nœud1 (simple) : {d_simple:.6f}")
    
    # Test 2: Distance vectorisée (A vers tous)
    d2_vectorized = _w2_dist2(X, xA, Y)
    d_vectorized = np.sqrt(d2_vectorized)
    print(f"Distances A->tous (vectorisé) : {d_vectorized}")
    
    # Test 3: Vérification cohérence
    print(f"\nVérification cohérence :")
    print(f"Simple A->nœud1 : {d_simple:.6f}")
    print(f"Vectorisé A->nœud1 : {d_vectorized[1]:.6f}")
    print(f"Différence : {abs(d_simple - d_vectorized[1]):.10f}")
    
    # Test 4: Performance
    import time
    N_large = 1000
    X_large = np.random.rand(N_large, 50).astype(np.float32)
    
    # Méthode simple (boucle)
    start = time.time()
    distances_simple = []
    for i in range(N_large):
        distances_simple.append(weighted_distance(xA, X_large[i], Y))
    time_simple = time.time() - start
    
    # Méthode vectorisée
    start = time.time()
    distances_vectorized = batch_weighted_distances(X_large, xA, Y)
    time_vectorized = time.time() - start
    
    print(f"\n=== Comparaison de performance (N={N_large}) ===")
    print(f"Méthode simple : {time_simple:.6f}s")
    print(f"Méthode vectorisée : {time_vectorized:.6f}s")
    print(f"Accélération : ×{time_simple/time_vectorized:.1f}")
