#!/usr/bin/env python3
"""
Module de chargement des données pour le projet Graph_Projet_A3MSI_5
Centralise toutes les fonctions de chargement et parsing des fichiers CSV

Fonctions disponibles :
- load_real_data() : Charge les données de nœuds depuis CSV (format simple)
- _load_graph_from_file() : Charge les données de nœuds (format robuste pour API)
- load_queries() : Charge les requêtes (format liste de dictionnaires)
- _load_queries_from_file() : Charge les requêtes (format DataFrame pour API)
"""

import numpy as np
import pandas as pd
import os
import re
from io import BytesIO

# ========== FONCTIONS DE build_graph.py ==========

def load_real_data(data_file="Data/adsSim_data_nodes.csv"):
    """
    Charge les données réelles depuis le fichier CSV.
    Retourne X (features), node_ids, et cluster_ids si disponibles.
    
    Format attendu : colonnes feature_1 à feature_50, node_id, cluster_id (optionnel)
    
    Returns:
        tuple: (X, node_ids, cluster_ids)
            - X : np.ndarray (N, 50) - matrice des features
            - node_ids : np.ndarray - identifiants des nœuds  
            - cluster_ids : np.ndarray ou None - identifiants des clusters
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
    
    Format attendu : colonnes point_A, Y_vector, D
    Y_vector au format "y1;y2;...;y50"
    
    Returns:
        list[dict]: Liste de requêtes avec clés 'point_A', 'Y_vector', 'D'
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

# ========== FONCTIONS DE apiameliore.py ==========

def _detect_id_col(df: pd.DataFrame) -> str:
    """
    Détecte automatiquement la colonne d'ID dans le CSV.
    Robuste aux différents noms de colonnes (node_id, id, node, etc.)
    """
    for c in ["node_id","id","ID","Id","node","Node","name","Name","point_A","A"]:
        if c in df.columns: return c
    for c in df.columns:
        if df[c].dtype == object: return c
    return df.columns[0]

def _load_graph_from_file(fobj):
    """
    Charge le CSV "graph" et renvoie (ids, X) de manière robuste.
    Version avancée pour l'API web avec détection automatique des formats.
    
    Supporte :
    - Format Y_vector : 50 dimensions encodées dans une colonne "y1;y2;...;y50"
    - Format feature_1..feature_50 : colonnes nommées explicitement
    - Format automatique : sélection des 50 colonnes numériques les plus variées
    
    Args:
        fobj: Objet fichier (BytesIO ou file-like)
        
    Returns:
        tuple: (ids, X)
            - ids : np.ndarray - identifiants des nœuds (string)
            - X : np.ndarray (N, 50) - matrice des features (float32)
    """
    df = pd.read_csv(fobj).dropna(axis=1, how="all")
    id_col = _detect_id_col(df)

    # Cas spécial: 50 dims encodées dans une colonne "Y_vector"
    if "Y_vector" in df.columns:
        ids = df[id_col].astype(str).values
        X = df["Y_vector"].astype(str).apply(
            lambda s: [float(x) for x in str(s).split(";") if x!=""]
        ).to_list()
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != 50:
            raise ValueError(f"Le graphe doit contenir 50 valeurs par nœud (trouvé {X.shape[1]}).")
        return ids, X

    # Sinon: tenter feature_1..feature_50, ou garder 50 colonnes numériques les plus variées
    feat_named = [c for c in df.columns if re.match(r"(?i)feature[_-]?\d+$", str(c))]
    if len(feat_named) >= 50:
        cols = sorted(feat_named, key=lambda x: int(re.findall(r"\d+", x)[0]))[:50]
        feats = df[cols]
    else:
        feats = df.drop(columns=list({id_col,"D","point_A","A"} & set(df.columns)))
        feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if feats.shape[1] > 50:
            variances = feats.var(axis=0, ddof=0)
            feats = feats[variances.sort_values(ascending=False).index[:50]]

    if feats.shape[1] != 50:
        raise ValueError(f"Après sélection : {feats.shape[1]} colonnes features (50 attendues).")

    ids = df[id_col].astype(str).values
    X = feats.to_numpy(dtype=np.float32)
    return ids, X

def _load_queries_from_file(fobj):
    """
    Charge le CSV "queries" et vérifie: point_A, Y_vector, D.
    Version pour l'API web avec validation stricte.
    
    Format attendu :
    - point_A : identifiant du nœud de départ
    - Y_vector : vecteur de pondération au format "y1;y2;...;y50"
    - D : rayon de recherche
    
    Args:
        fobj: Objet fichier (BytesIO ou file-like)
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes point_A, Y_vector, D validées
    """
    qdf = pd.read_csv(fobj)
    need = {"point_A","Y_vector","D"}
    if not need.issubset(set(qdf.columns)):
        raise ValueError("Le fichier queries doit contenir : point_A, Y_vector, D")
    qdf["point_A"] = qdf["point_A"].astype(str)
    qdf["D"] = pd.to_numeric(qdf["D"], errors="coerce").astype(float)
    if qdf["D"].isna().any(): 
        raise ValueError("Colonne D invalide.")
    return qdf

# ========== FONCTIONS UTILITAIRES ==========

def _parse_y(s: str) -> np.ndarray:
    """
    Parse "y1;...;y50" en array(50,). 
    Validation stricte : erreur si la taille != 50.
    
    Args:
        s: String au format "y1;y2;...;y50"
        
    Returns:
        np.ndarray: Array de 50 coefficients (float32)
    """
    arr = np.asarray([float(x) for x in str(s).split(";") if x!=""], dtype=np.float32)
    if arr.size != 50:
        raise ValueError(f"Y_vector doit contenir 50 coefficients (reçu {arr.size}).")
    return arr

def _graph_prefix(ids: np.ndarray) -> str:
    """
    Extrait le préfixe des identifiants de nœuds (ex: node_, ads_).
    """
    s = str(ids[0])
    return s.split("_",1)[0] if "_" in s else "node"

def _map_pointA_to_graph_prefix(qdf: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """
    Aligne le préfixe des ids si nécessaire (ex: ads_1 -> node_1) 
    pour éviter les mismatches entre requêtes et graphe.
    """
    if qdf["point_A"].isin(ids).all(): 
        return qdf
    prefix = _graph_prefix(ids)
    q = qdf.copy()
    q["point_A"] = q["point_A"].astype(str).apply(
        lambda s: re.sub(r"^[A-Za-z]+_", f"{prefix}_", s)
    )
    return q
