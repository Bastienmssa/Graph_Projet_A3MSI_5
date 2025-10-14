# weighted_distance.py
"""
Module pour l'implémentation de la distance pondérée.
Contient toutes les fonctions liées à l'étape 2 du projet : Implémentation de la distance pondérée.

Formule implémentée : d_Y(u,v) = √(Σ(k=1 to 50) y_k(f_uk - f_vk)²)
"""

import numpy as np

def weighted_squared_distance(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcule la distance pondérée au carré entre xA et tous les points de X.
    
    Implémente la formule : d_Y²(u,v) = Σ(k=1 to 50) y_k(f_uk - f_vk)²
    
    Args:
        X: Matrice des points (N, 50) - tous les nœuds du graphe
        xA: Point de référence (50,) - nœud de départ
        y: Vecteur de poids (50,) - pondération Y
        
    Returns:
        np.ndarray: Distances au carré pondérées (N,) pour chaque point de X
        
    Example:
        >>> X = np.array([[1, 2, 3], [4, 5, 6]])  # 2 points, 3 features
        >>> xA = np.array([0, 0, 0])              # point de référence
        >>> y = np.array([1, 1, 1])               # poids uniformes
        >>> d2 = weighted_squared_distance(X, xA, y)
        >>> print(d2)  # [14, 77] = [1²+2²+3², 4²+5²+6²]
    """
    diff = X - xA  # Différence vectorisée (N, 50)
    return (diff * diff) @ y  # Produit matriciel efficace

def weighted_distance(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcule la distance pondérée finale entre xA et tous les points de X.
    
    Implémente la formule complète : d_Y(u,v) = √(Σ(k=1 to 50) y_k(f_uk - f_vk)²)
    
    Args:
        X: Matrice des points (N, 50)
        xA: Point de référence (50,)
        y: Vecteur de poids (50,)
        
    Returns:
        np.ndarray: Distances pondérées finales (N,)
    """
    d2 = weighted_squared_distance(X, xA, y)
    return np.sqrt(d2)

def knn_weighted_distance(idx: int, X: np.ndarray, y: np.ndarray, k: int):
    """
    Trouve les k plus proches voisins d'un nœud selon la distance pondérée.
    
    Args:
        idx: Index du nœud de référence dans X
        X: Matrice des points (N, 50)
        y: Vecteur de poids (50,)
        k: Nombre de voisins à retourner
        
    Returns:
        tuple: (indices des k voisins, distances correspondantes)
        
    Note:
        Le nœud lui-même (idx) est exclu des résultats.
    """
    xA = X[idx]
    d2 = weighted_squared_distance(X, xA, y)
    d2[idx] = np.inf  # Exclure le nœud lui-même
    k = min(k, X.shape[0] - 1)  # Ne pas dépasser le nombre de nœuds disponibles
    
    # Sélection efficace des k plus petites distances
    nn_idx = np.argpartition(d2, k)[:k]
    order = np.argsort(d2[nn_idx], kind="stable")  # Tri stable pour la reproductibilité
    nn_idx = nn_idx[order]
    
    return nn_idx, np.sqrt(d2[nn_idx])

def validate_weighted_distance_formula():
    """
    Fonction de test pour valider l'implémentation de la distance pondérée.
    
    Returns:
        bool: True si tous les tests passent
    """
    # Test 1: Cas simple avec 3 features
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    y = np.array([0.5, 1.0, 2.0])
    
    # Calcul manuel : d_Y^2 = 0.5*(1-4)^2 + 1.0*(2-5)^2 + 2.0*(3-6)^2
    expected_d2 = 0.5 * (1-4)**2 + 1.0 * (2-5)**2 + 2.0 * (3-6)**2
    
    # Test avec notre fonction
    X = np.array([v])
    result_d2 = weighted_squared_distance(X, u, y)[0]
    
    test1_pass = np.isclose(expected_d2, result_d2)
    
    # Test 2: Distance nulle avec le même point
    result_d2_same = weighted_squared_distance(np.array([u]), u, y)[0]
    test2_pass = np.isclose(result_d2_same, 0.0)
    
    # Test 3: Symétrie (d(u,v) = d(v,u))
    d_uv = weighted_distance(np.array([v]), u, y)[0]
    d_vu = weighted_distance(np.array([u]), v, y)[0]
    test3_pass = np.isclose(d_uv, d_vu)
    
    print(f"Test 1 (formule): {'✅' if test1_pass else '❌'}")
    print(f"Test 2 (distance nulle): {'✅' if test2_pass else '❌'}")
    print(f"Test 3 (symétrie): {'✅' if test3_pass else '❌'}")
    
    return test1_pass and test2_pass and test3_pass

if __name__ == "__main__":
    # Tests automatiques
    print("🧪 Tests de validation de la distance pondérée:")
    all_tests_pass = validate_weighted_distance_formula()
    print(f"\n{'🎉 Tous les tests passent!' if all_tests_pass else '❌ Certains tests échouent'}")
