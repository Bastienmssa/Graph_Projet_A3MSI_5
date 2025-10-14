# graph_build.py
"""
Module pour la construction et le chargement des graphes.
Contient toutes les fonctions liées à l'étape 1 du projet : Construction du graphe.
"""

import re
import numpy as np
import pandas as pd
from io import BytesIO
from fastapi import UploadFile

# Import des fonctions de distance pondérée
from weighted_distance import weighted_squared_distance, knn_weighted_distance

# ========= Chargement & colonnes =========

def read_csv_flex(up: UploadFile) -> bytes:
    """Lit un fichier CSV uploadé et retourne les bytes bruts."""
    return up.file.read()

def detect_id_col(df: pd.DataFrame) -> str:
    """
    Détecte automatiquement la colonne d'identifiants dans un DataFrame.
    Cherche d'abord des noms standards, puis la première colonne de type object.
    """
    for c in ["node_id","id","ID","Id","node","Node","name","Name","point_A","A"]:
        if c in df.columns: 
            return c
    for c in df.columns:
        if df[c].dtype == object: 
            return c
    return df.columns[0]

def load_graph_from_csv_bytes(csv_bytes: bytes):
    """
    Charge un graphe depuis des bytes CSV.
    
    Args:
        csv_bytes: Données CSV en bytes
        
    Returns:
        tuple: (ids, X) où ids sont les identifiants des nœuds 
               et X est la matrice (N, 50) des features
               
    Raises:
        ValueError: Si le graphe n'a pas exactement 50 features par nœud
    """
    df = pd.read_csv(BytesIO(csv_bytes)).dropna(axis=1, how="all")
    id_col = detect_id_col(df)

    # Cas 1: Y_vector déjà présent (format "y1;y2;...;y50")
    if "Y_vector" in df.columns:
        ids = df[id_col].astype(str).values
        X = df["Y_vector"].astype(str).apply(
            lambda s: [float(x) for x in str(s).split(";") if x != ""]
        ).to_list()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 50:
            raise ValueError("Le graphe doit avoir 50 valeurs par nœud (50 dims).")
        return ids, X

    # Cas 2: features colonnes numériques (on en garde 50)
    feat_named = [c for c in df.columns if re.match(r"(?i)feature[_-]?\d+$", str(c))]
    if len(feat_named) >= 50:
        # Tri par numéro de feature
        cols = sorted(feat_named, key=lambda x: int(re.findall(r"\d+", x)[0]))[:50]
        feats = df[cols]
    else:
        # Prendre toutes les colonnes numériques sauf les colonnes spéciales
        feats = df.drop(columns=list({id_col,"D","point_A","A","Y_vector"} & set(df.columns)))
        feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if feats.shape[1] > 50:
            # Garder les 50 features avec la plus grande variance
            variances = feats.var(axis=0, ddof=0)
            feats = feats[variances.sort_values(ascending=False).index[:50]]
    
    if feats.shape[1] != 50:
        raise ValueError(f"Après sélection: {feats.shape[1]} colonnes features (50 attendues).")
    
    ids = df[id_col].astype(str).values
    X = feats.to_numpy(dtype=np.float32)
    return ids, X

def parse_y_vector(s: str) -> np.ndarray:
    """
    Parse un Y_vector au format "y1;y2;...;y50".
    
    Args:
        s: String au format "y1;y2;...;y50"
        
    Returns:
        np.ndarray: Array de 50 valeurs float32
        
    Raises:
        ValueError: Si le nombre de valeurs n'est pas exactement 50
    """
    arr = np.asarray([float(x) for x in str(s).split(";") if x != ""], dtype=np.float32)
    if arr.shape != (50,): 
        raise ValueError("Y_vector doit contenir exactement 50 valeurs.")
    return arr

def default_y_from_queries_csv(qfile_bytes: bytes | None) -> np.ndarray:
    """
    Extrait un Y_vector par défaut depuis un fichier queries CSV.
    
    Args:
        qfile_bytes: Bytes du fichier queries CSV, ou None
        
    Returns:
        np.ndarray: Y_vector par défaut (premier du CSV ou vecteur uniforme)
    """
    if not qfile_bytes: 
        return np.ones(50, dtype=np.float32)
    
    try:
        qdf = pd.read_csv(BytesIO(qfile_bytes))
        if "Y_vector" in qdf.columns and len(qdf) > 0:
            y = parse_y_vector(str(qdf.iloc[0]["Y_vector"]))
            return np.asarray(y, dtype=np.float32)
    except Exception:
        pass
    
    return np.ones(50, dtype=np.float32)

def extract_queries_cols(qdf: pd.DataFrame):
    """
    Extrait et valide les colonnes nécessaires d'un DataFrame queries.
    
    Args:
        qdf: DataFrame des queries
        
    Returns:
        str: Nom de la colonne contenant les points A
        
    Raises:
        ValueError: Si les colonnes requises sont absentes
    """
    colA = "point_A" if "point_A" in qdf.columns else ("A" if "A" in qdf.columns else None)
    if colA is None: 
        raise ValueError("Queries: colonne 'point_A' (ou 'A') absente.")
    if "D" not in qdf.columns: 
        raise ValueError("Queries: colonne 'D' absente.")
    return colA

# ========= Normalisation d'IDs =========

_num_pat = re.compile(r"(\d+)$")
_prefix_pat = re.compile(r"^(\D*?)(\d+)$")

def infer_graph_prefix(ids: np.ndarray) -> str:
    """
    Déduit le préfixe non-numérique des IDs du graphe.
    
    Args:
        ids: Array des identifiants du graphe
        
    Returns:
        str: Préfixe commun (ex: 'node_' pour 'node_123')
    """
    s = str(ids[0])
    m = _prefix_pat.match(s)
    return m.group(1) if m else ""

def normalize_node_id(node_id: str, id_to_idx: dict, graph_prefix: str) -> str | None:
    """
    Normalise un ID de nœud pour qu'il corresponde au format du graphe.
    
    Args:
        node_id: ID à normaliser
        id_to_idx: Dictionnaire de mapping ID -> index
        graph_prefix: Préfixe du graphe
        
    Returns:
        str | None: ID normalisé s'il existe, None sinon
    """
    # Si déjà présent → OK
    if node_id in id_to_idx: 
        return node_id
    
    # Sinon, on prend la partie numérique finale et on reconstruit prefix+digits
    m = _num_pat.search(node_id)
    if not m: 
        return None
    
    candidate = f"{graph_prefix}{m.group(1)}"
    if candidate in id_to_idx: 
        return candidate
    
    return None

# ========= Aliases pour compatibilité =========

# Aliases pour maintenir la compatibilité avec le code existant
knn_of_node = knn_weighted_distance
