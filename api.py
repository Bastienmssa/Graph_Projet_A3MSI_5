# api.py
"""
API REST pour le TP Graph — Recherche & Bonus.
Contient tous les endpoints pour la recherche pondérée et les algorithmes de chemins.
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os, re, heapq, time
import numpy as np
import pandas as pd
from io import BytesIO, StringIO

# Import des modules du projet
from graph_build import (
    read_csv_flex, detect_id_col, load_graph_from_csv_bytes,
    parse_y_vector, default_y_from_queries_csv, extract_queries_cols,
    infer_graph_prefix, normalize_node_id, knn_of_node
)

from weighted_distance import weighted_squared_distance

from radiusx_search import (
    radius_search_naive, radius_search_pruned, compare_radius_search_strategies,
    radius_search_to_csv
)

# Configuration de l'application
app = FastAPI(
    title="TP Graph — Recherche & Bonus API", 
    version="8.0",
    description="API pour la recherche pondérée dans les graphes et les algorithmes de chemins optimaux"
)

# Servir les fichiers statiques du frontend
if os.path.exists("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# ========= Aliases pour compatibilité avec le code existant =========

_read_csv_flex = read_csv_flex
_detect_id_col = detect_id_col
_load_graph_from_csv_bytes = load_graph_from_csv_bytes
_parse_y = parse_y_vector
_default_y_from_queries_csv = default_y_from_queries_csv
_extract_queries_cols = extract_queries_cols
_infer_graph_prefix = infer_graph_prefix
_normalize_A = normalize_node_id
_w2_dist2 = weighted_squared_distance
_knn_of = knn_of_node

# ========= Endpoints de recherche dans le rayon =========

@app.post("/search_csv")
def search_csv(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    mode: str = Query(default="pruned"),              # "naive" | "pruned"
    top_m: int = Query(default=12, ge=1, le=50),
    dscale: float = Query(default=1.0, ge=0.05, le=2.0),
):
    """
    Recherche dans le rayon X avec différentes stratégies.
    
    Args:
        graph_file: Fichier CSV du graphe
        queries_file: Fichier CSV des requêtes
        mode: Mode de recherche ("naive" ou "pruned")
        top_m: Nombre de features pour la borne inférieure (mode pruned)
        dscale: Facteur d'échelle pour le rayon
        
    Returns:
        CSV des résultats avec métadonnées dans les headers
    """
    try:
        gbytes = _read_csv_flex(graph_file)
        qbytes = _read_csv_flex(queries_file)
        ids, X = _load_graph_from_csv_bytes(gbytes)
        qdf = pd.read_csv(BytesIO(qbytes))
        colA = _extract_queries_cols(qdf)

        if mode == "naive":
            rows, t, matched, skipped = radius_search_naive(ids, X, qdf, colA, dscale)
            pruned_rejects = 0
        elif mode == "pruned":
            rows, t, pruned_rejects, matched, skipped = radius_search_pruned(ids, X, qdf, colA, dscale, top_m)
        else:
            raise ValueError("mode doit être 'naive' ou 'pruned'.")

        # CSV de sortie
        csv_content = radius_search_to_csv(rows, mode)

        headers = {
            "X-Alg-Time": f"{t:.6f}",
            "X-Row-Count": str(len(rows)),
            "X-PrunedRejects": str(pruned_rejects),
            "X-Matched": str(matched),
            "X-Skipped": str(skipped),
            "Content-Disposition": f'attachment; filename="{mode}_results.csv"',
        }
        return StreamingResponse(iter([csv_content]), media_type="text/csv", headers=headers)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/compare_search")
def compare_search(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    top_m: int = Query(default=12, ge=1, le=50),
    dscale: float = Query(default=1.0, ge=0.05, le=2.0),
):
    """
    Compare les performances des stratégies naïve vs structurée.
    
    Returns:
        JSON avec les statistiques de comparaison
    """
    try:
        gbytes = _read_csv_flex(graph_file)
        qbytes = _read_csv_flex(queries_file)
        ids, X = _load_graph_from_csv_bytes(gbytes)
        qdf = pd.read_csv(BytesIO(qbytes))
        colA = _extract_queries_cols(qdf)

        comparison_results = compare_radius_search_strategies(ids, X, qdf, colA, dscale, top_m)
        return comparison_results
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ========= Algorithmes de chemins (A* et Beam Search) =========

def _astar_path(ids, X, y, src_id: str, dst_id: str, k: int = 10, max_expansions: int = 20000):
    """
    Algorithme A* pour trouver le chemin optimal entre deux nœuds.
    
    Args:
        ids: Identifiants des nœuds
        X: Matrice des features
        y: Vecteur de pondération
        src_id: Nœud source
        dst_id: Nœud destination
        k: Nombre de voisins k-NN
        max_expansions: Limite d'expansions
        
    Returns:
        dict: Résultats avec chemin, coût, statistiques
    """
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    if src_id not in id_to_idx or dst_id not in id_to_idx:
        raise ValueError("A ou B introuvable dans le graphe.")
    s = id_to_idx[src_id]; t = id_to_idx[dst_id]
    X32 = X.astype(np.float32, copy=False)

    def h(idx):
        diff = X32[idx] - X32[t]
        return float(np.sqrt((diff*diff) @ y))

    openq = []
    heapq.heappush(openq, (h(s), 0.0, s))
    parents = {s: -1}
    best_g = {s: 0.0}
    expanded = 0

    while openq and expanded < max_expansions:
        f, g, u = heapq.heappop(openq)
        expanded += 1
        if u == t:
            path = []
            cur = u
            while cur != -1:
                path.append(ids[cur]); cur = parents.get(cur, -1)
            path.reverse()
            return {"path": path, "cost": g, "expanded": expanded, "k": k, "exact": True}

        nn_idx, nn_d = _knn_of(u, X32, y, k)
        for v, w in zip(nn_idx, nn_d):
            g2 = g + float(w)
            if v not in best_g or g2 < best_g[v] - 1e-12:
                best_g[v] = g2
                parents[v] = u
                heapq.heappush(openq, (g2 + h(v), g2, v))

    return {"path": [], "cost": None, "expanded": expanded, "k": k, "exact": True, "note": "max expansions atteint"}

def _beam_k_path(ids, X, y, src_id: str, dst_id: str, k_neighbors: int = 10, K: int = 6, beam_width: int = 16):
    """
    Algorithme Beam Search pour trouver un chemin approximatif.
    
    Args:
        ids: Identifiants des nœuds
        X: Matrice des features
        y: Vecteur de pondération
        src_id: Nœud source
        dst_id: Nœud destination
        k_neighbors: Nombre de voisins k-NN
        K: Nombre d'étapes
        beam_width: Largeur du faisceau
        
    Returns:
        dict: Résultats avec chemin, coût, statistiques
    """
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    if src_id not in id_to_idx or dst_id not in id_to_idx:
        raise ValueError("A ou B introuvable dans le graphe.")
    s = id_to_idx[src_id]; t = id_to_idx[dst_id]
    X32 = X.astype(np.float32, copy=False)

    def h(idx):
        diff = X32[idx] - X32[t]
        return float(np.sqrt((diff*diff) @ y))

    beam = [(h(s), 0.0, s, [ids[s]])]  # (score, g, node_idx, path)
    best_goal = None
    expanded = 0

    for _ in range(K):
        next_candidates = []
        for score, g, u, path in beam:
            nn_idx, nn_d = _knn_of(u, X32, y, k_neighbors)
            expanded += 1
            for v, w in zip(nn_idx, nn_d):
                if ids[v] in path:  # éviter cycles simples
                    continue
                g2 = g + float(w)
                path2 = path + [ids[v]]
                if v == t:
                    if best_goal is None or g2 < best_goal[0]:
                        best_goal = (g2, path2)
                else:
                    next_candidates.append((g2 + h(v), g2, v, path2))
        if not next_candidates:
            break
        next_candidates.sort(key=lambda x: x[0])
        beam = next_candidates[:beam_width]
        if best_goal is not None:
            return {"path": best_goal[1], "cost": best_goal[0], "expanded": expanded,
                    "k_neighbors": k_neighbors, "K": K, "beam": beam_width, "exact": False}
    if best_goal is not None:
        return {"path": best_goal[1], "cost": best_goal[0], "expanded": expanded,
                "k_neighbors": k_neighbors, "K": K, "beam": beam_width, "exact": False}
    if beam:
        best = min(beam, key=lambda x: x[0])
        return {"path": best[3], "cost": best[1], "expanded": expanded,
                "k_neighbors": k_neighbors, "K": K, "beam": beam_width, "exact": False, "note": "but non atteint (meilleur partiel)"}
    return {"path": [], "cost": None, "expanded": expanded,
            "k_neighbors": k_neighbors, "K": K, "beam": beam_width, "exact": False, "note": "aucun chemin généré"}

# ========= Endpoints pour les algorithmes de chemins =========

@app.post("/path_astar")
def path_astar(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile | None = File(default=None),
    src: str = Query(...), dst: str = Query(...),
    y_vector: str | None = Query(default=None),
    k: int = Query(default=10, ge=2, le=200),
    maxexp: int = Query(default=20000, ge=100, le=1000000),
):
    """
    Calcul de chemin optimal avec l'algorithme A*.
    
    Returns:
        JSON avec le chemin, coût et statistiques
    """
    try:
        gbytes = _read_csv_flex(graph_file)
        ids, X = _load_graph_from_csv_bytes(gbytes)

        qbytes = _read_csv_flex(queries_file) if queries_file is not None else None
        if y_vector is not None and len(y_vector.strip()) > 0:
            y = _parse_y(y_vector)
        else:
            y = _default_y_from_queries_csv(qbytes)
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (50,): raise ValueError("Y_vector doit contenir 50 valeurs.")

        t0 = time.perf_counter()
        res = _astar_path(ids, X, y, src, dst, k=k, max_expansions=maxexp)
        res.update({"n_nodes": len(ids), "algo_time_sec": time.perf_counter()-t0})
        return res
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/path_beam")
def path_beam(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile | None = File(default=None),
    src: str = Query(...), dst: str = Query(...),
    y_vector: str | None = Query(default=None),
    k_neighbors: int = Query(default=10, ge=2, le=200),
    K: int = Query(default=6, ge=1, le=100),
    beam: int = Query(default=16, ge=2, le=200),
):
    """
    Calcul de chemin approximatif avec l'algorithme Beam Search.
    
    Returns:
        JSON avec le chemin, coût et statistiques
    """
    try:
        gbytes = _read_csv_flex(graph_file)
        ids, X = _load_graph_from_csv_bytes(gbytes)

        qbytes = _read_csv_flex(queries_file) if queries_file is not None else None
        if y_vector is not None and len(y_vector.strip()) > 0:
            y = _parse_y(y_vector)
        else:
            y = _default_y_from_queries_csv(qbytes)
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (50,): raise ValueError("Y_vector doit contenir 50 valeurs.")

        t0 = time.perf_counter()
        res = _beam_k_path(ids, X, y, src, dst, k_neighbors=k_neighbors, K=K, beam_width=beam)
        res.update({"n_nodes": len(ids), "algo_time_sec": time.perf_counter()-t0})
        return res
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ========= Interface web =========

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Page d'accueil avec l'interface utilisateur.
    
    Returns:
        HTML de l'interface web
    """
    # Vérifier si le fichier template existe
    template_path = "frontend/templates/index.html"
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Fallback simple si le template n'existe pas
        return """
        <html>
        <head><title>TP Graph API</title></head>
        <body>
            <h1>TP Graph — Recherche & Bonus API</h1>
            <p>Interface web non disponible. Utilisez les endpoints API directement :</p>
            <ul>
                <li><code>POST /search_csv</code> - Recherche dans le rayon</li>
                <li><code>POST /compare_search</code> - Comparaison des stratégies</li>
                <li><code>POST /path_astar</code> - Algorithme A*</li>
                <li><code>POST /path_beam</code> - Algorithme Beam Search</li>
            </ul>
            <p><a href="/docs">Documentation API (Swagger)</a></p>
        </body>
        </html>
        """
