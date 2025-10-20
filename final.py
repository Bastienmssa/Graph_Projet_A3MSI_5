# api.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn, os, re, heapq, time
import numpy as np
import pandas as pd
from io import BytesIO, StringIO

app = FastAPI(title="TP Graph — Recherche & Chemins (complet)", version="9.1")

# ======================================================================
# =========================  UTILITAIRES CSV  ===========================
# ======================================================================

def _read_csv(up: UploadFile) -> bytes:
    return up.file.read()

def _detect_id_col(df: pd.DataFrame) -> str:
    # Heuristique simple pour la colonne ID
    for c in ["node_id","id","ID","Id","node","Node","name","Name","point_A","A"]:
        if c in df.columns: return c
    for c in df.columns:
        if df[c].dtype==object: return c
    return df.columns[0]

def _load_graph(csv_bytes: bytes):
    """
    Charge le graph au sens 'données':
      - ids: np.ndarray[str] des identifiants de nœuds
      - X  : np.ndarray float32 (N,50) des 50 features
    """
    df = pd.read_csv(BytesIO(csv_bytes)).dropna(axis=1, how="all")
    id_col = _detect_id_col(df)

    # Cas colonnes nommées feature_1..feature_50
    feat_named = [c for c in df.columns if re.match(r"(?i)feature[_-]?\d+$", str(c))]
    if len(feat_named) >= 50:
        cols = sorted(feat_named, key=lambda x:int(re.findall(r"\d+", x)[0]))[:50]
        feats = df[cols]
    else:
        # Sinon on garde 50 colonnes numériques les + variées
        feats = df.drop(columns=[id_col], errors="ignore")
        feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if feats.shape[1] > 50:
            variances = feats.var(axis=0, ddof=0)
            feats = feats[variances.sort_values(ascending=False).index[:50]]

    if feats.shape[1] != 50:
        raise ValueError(f"Le graph doit avoir 50 features par nœud (trouvé {feats.shape[1]}).")

    ids = df[id_col].astype(str).values
    X = feats.to_numpy(dtype=np.float32)
    return ids, X

def _load_queries(csv_bytes: bytes) -> pd.DataFrame:
    """
    Accepte 2 formats:
      - point_A, A_vector(50 ';'), Y_vector(50 ';'), D    (A = nouveau point)
      - point_A,               Y_vector(50 ';'), D        (A = nœud du graph)
    """
    q = pd.read_csv(BytesIO(csv_bytes))
    need_min = {"point_A","Y_vector","D"}
    if not need_min.issubset(q.columns):
        raise ValueError("Queries CSV doit contenir au minimum: point_A, Y_vector, D (et A_vector si A n'est pas dans le graph).")
    return q

def _parse_vec50(cell: str, name: str) -> np.ndarray:
    v = np.asarray([float(x) for x in str(cell).split(";") if x!=""], dtype=np.float32)
    if v.shape != (50,):
        raise ValueError(f"{name} doit contenir exactement 50 valeurs séparées par ';'.")
    return v

# ======================================================================
# ============== CONSTRUCTION D’UN “GRAPHE” (léger, demandé) ===========
# ======================================================================

def construct_graph(ids: np.ndarray, X: np.ndarray) -> dict:
    """
    Construit une structure légère (sans arêtes) à partir du CSV:
      - 'ids'       : liste des IDs
      - 'X'         : matrice N×50 des features
      - 'id_to_idx' : dict ID -> index
    Appelée systématiquement au chargement du graph.
    """
    return {
        "ids": ids,
        "X": X,
        "id_to_idx": {ids[i]: i for i in range(len(ids))}
    }

# ======================================================================
# ==========================  DISTANCES  ================================
# ======================================================================

def _w2_dist2_all(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d² pondérée entre A et tous les nœuds: ( (X - xA)^2 ) @ y."""
    diff = X - xA
    return (diff*diff) @ y

# ======================================================================
# =======================  NORMALISATION D’ID  ==========================
# ======================================================================

_num_pat    = re.compile(r"(\d+)$")
_prefix_pat = re.compile(r"^(\D*?)(\d+)$")

def _infer_graph_prefix(ids: np.ndarray) -> str:
    s = str(ids[0]); m = _prefix_pat.match(s)
    return m.group(1) if m else ""

def _normalize_to_graph(A: str, id_to_idx: dict, graph_prefix: str) -> str | None:
    """
    Si A n'est pas dans le graph et se termine par des chiffres, on tente graph_prefix+digits.
    (Compatibilité si le prof fournit A = node_#### et le graph = node_####).
    Si A est un nouveau point (ads_* ou autre) SANS A_vector, on ne peut pas calculer.
    """
    if A in id_to_idx: return A
    m = _num_pat.search(A)
    if not m: return None
    cand = f"{graph_prefix}{m.group(1)}"
    return cand if cand in id_to_idx else None

# ======================================================================
# ======================  RECHERCHE NAÏF / PRUNED  =====================
# ======================================================================

def _solve_one_row(ids, X, row, graph, dscale: float, mode: str, top_m: int):
    """
    Résout une requête et renvoie une liste triée [(node_id, dist), ...].
    - Si 'A_vector' présent: A est un nouveau point (pas dans le graph).
    - Sinon: A doit exister dans le graph (compat prof).
    """
    y  = _parse_vec50(row["Y_vector"], "Y_vector")
    D  = float(row["D"])
    Deff2 = (D * dscale) ** 2

    # ------- Cas 1: A est un nouveau point (A_vector fourni) -------
    if "A_vector" in row and isinstance(row["A_vector"], str) and row["A_vector"].strip():
        xA = _parse_vec50(row["A_vector"], "A_vector")

        if mode == "naive":
            d2 = _w2_dist2_all(X, xA, y)
            sel = np.where(d2 <= Deff2)[0]
            if sel.size == 0: return []
            d = np.sqrt(d2[sel]); ord_ = np.argsort(d, kind="stable")
            return [(ids[j], float(di)) for j, di in zip(sel[ord_], d[ord_])]

        # pruned Top-M
        m = max(1, min(int(top_m), 50))
        top = np.argsort(-y)[:m]
        diffM = X[:, top] - xA[top]
        lb2 = (diffM*diffM) @ y[top]
        cand = np.where(lb2 <= Deff2)[0]
        if cand.size == 0: return []
        d2 = _w2_dist2_all(X[cand], xA, y)
        mask = d2 <= Deff2
        cand2 = cand[mask]
        if cand2.size == 0: return []
        d = np.sqrt(d2[mask]); ord_ = np.argsort(d, kind="stable")
        return [(ids[j], float(di)) for j, di in zip(cand2[ord_], d[ord_])]

    # ------- Cas 2: pas d’A_vector -> A doit exister dans le graph -------
    Araw = str(row["point_A"])
    A = _normalize_to_graph(Araw, graph["id_to_idx"], _infer_graph_prefix(graph["ids"]))
    if A is None:
        # Requête inexploitable: A n'est pas dans le graph et pas d'A_vector
        return []
    idxA = graph["id_to_idx"][A]
    xA = graph["X"][idxA]

    if mode == "naive":
        d2 = _w2_dist2_all(X, xA, y); d2[idxA] = np.inf
        sel = np.where(d2 <= Deff2)[0]
        if sel.size == 0: return []
        d = np.sqrt(d2[sel]); ord_ = np.argsort(d, kind="stable")
        return [(ids[j], float(di)) for j, di in zip(sel[ord_], d[ord_])]

    # pruned Top-M
    m = max(1, min(int(top_m), 50))
    top = np.argsort(-y)[:m]
    diffM = X[:, top] - xA[top]
    lb2 = (diffM*diffM) @ y[top]; lb2[idxA] = np.inf
    cand = np.where(lb2 <= Deff2)[0]
    if cand.size == 0: return []
    d2 = _w2_dist2_all(X[cand], xA, y)
    mask = d2 <= Deff2
    cand2 = cand[mask]
    if cand2.size == 0: return []
    d = np.sqrt(d2[mask]); ord_ = np.argsort(d, kind="stable")
    return [(ids[j], float(di)) for j, di in zip(cand2[ord_], d[ord_])]

# ======================================================================
# ==========================  ENDPOINTS RECHERCHE  =====================
# ======================================================================

@app.post("/search_csv")
def search_csv(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    mode: str = Query(default="pruned"),              # "naive" | "pruned"
    top_m: int = Query(default=12, ge=1, le=50),
    dscale: float = Query(default=1.0, ge=0.05, le=2.0),
):
    """
    Résultats détaillés 'long' (plusieurs lignes par requête):
    query_idx,point_A,node,dist
    """
    try:
        ids, X = _load_graph(_read_csv(graph_file))
        qdf = _load_queries(_read_csv(queries_file))
        graph = construct_graph(ids, X)

        rows = []
        t0 = time.perf_counter()
        for qi, r in qdf.iterrows():
            pairs = _solve_one_row(ids, X, r, graph, dscale, mode, top_m)
            for nid, dist in pairs:
                rows.append((qi, str(r["point_A"]), nid, dist))
        algo_time = time.perf_counter() - t0

        buf = StringIO(); buf.write("query_idx,point_A,node,dist\n")
        for qi, A, nid, dist in rows:
            buf.write(f"{qi},{A},{nid},{dist:.6f}\n")
        headers = {
            "X-Alg-Time": f"{algo_time:.6f}",
            "X-Row-Count": str(len(rows)),
            "Content-Disposition": f'attachment; filename=\"{mode}_results.csv\"',
        }
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/search_summary_csv")
def search_summary_csv(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    mode: str = Query(default="pruned"),              # "naive" | "pruned"
    top_m: int = Query(default=12, ge=1, le=50),
    dscale: float = Query(default=1.0, ge=0.05, le=2.0),
):
    """
    Résumé EXACTEMENT 1 ligne par requête:
    query_id,D,num_matches,nodes,nodes_with_distance
    """
    try:
        ids, X = _load_graph(_read_csv(graph_file))
        qdf = _load_queries(_read_csv(queries_file))
        graph = construct_graph(ids, X)

        out = StringIO()
        out.write("query_id,D,num_matches,nodes,nodes_with_distance\n")

        t0 = time.perf_counter()
        for _, r in qdf.iterrows():
            qid = str(r["point_A"]); D = float(r["D"])
            pairs = _solve_one_row(ids, X, r, graph, dscale, mode, top_m)
            if not pairs:
                out.write(f"{qid},{D:.6f},0,,\n")
            else:
                nodes = ";".join(p[0] for p in pairs)
                nodes_d = ";".join(f"{p[0]}:{p[1]:.6f}" for p in pairs)
                out.write(f"{qid},{D:.6f},{len(pairs)},{nodes},{nodes_d}\n")
        algo_time = time.perf_counter() - t0

        headers = {"X-Alg-Time": f"{algo_time:.6f}",
                   "Content-Disposition": f'attachment; filename=\"summary_{mode}.csv\"'}
        return StreamingResponse(iter([out.getvalue()]), media_type="text/csv", headers=headers)
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
    Compare naïf vs pruned sur les mêmes requêtes:
    renvoie temps (s) et égalité des résultats détaillés.
    """
    try:
        ids, X = _load_graph(_read_csv(graph_file))
        qdf = _load_queries(_read_csv(queries_file))
        graph = construct_graph(ids, X)

        # naïf
        t0 = time.perf_counter(); N=[]
        for _, r in qdf.iterrows():
            pairs = _solve_one_row(ids, X, r, graph, dscale, "naive", top_m)
            N.extend([(str(r["point_A"]), nid, dist) for nid, dist in pairs])
        tN = time.perf_counter()-t0

        # pruned
        t0 = time.perf_counter(); P=[]
        for _, r in qdf.iterrows():
            pairs = _solve_one_row(ids, X, r, graph, dscale, "pruned", top_m)
            P.extend([(str(r["point_A"]), nid, dist) for nid, dist in pairs])
        tP = time.perf_counter()-t0

        equal = (sorted(N) == sorted(P))
        return {"naive_time_sec":tN,"pruned_time_sec":tP,"rows_naive":len(N),"rows_pruned":len(P),"equal":equal}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ======================================================================
# ==============================  BONUS  ===============================
# ==========  A* exact et Beam K-sauts (k-NN à la volée sur Y)  ========
# ======================================================================

def _w2_dists_from_idx(X32: np.ndarray, y: np.ndarray, idx: int):
    diff = X32 - X32[idx]
    d2 = (diff*diff) @ y
    d2[idx] = np.inf
    return np.sqrt(d2)

def _knn_of(idx: int, X32: np.ndarray, y: np.ndarray, k: int):
    d = _w2_dists_from_idx(X32, y, idx)
    k = min(k, len(d)-1)
    idxs = np.argpartition(d, k)[:k]
    ord_ = np.argsort(d[idxs], kind="stable")
    return idxs[ord_], d[idxs][ord_]

def _astar_path(ids, X, y, src_id: str, dst_id: str, k: int = 10, max_expansions: int = 20000):
    id2 = {ids[i]: i for i in range(len(ids))}
    if src_id not in id2 or dst_id not in id2:
        raise ValueError("A ou B introuvable dans le graph.")
    s, t = id2[src_id], id2[dst_id]
    X32 = X.astype(np.float32, copy=False)

    def h(i):
        diff = X32[i] - X32[t]
        return float(np.sqrt((diff*diff) @ y))

    openq = [(h(s), 0.0, s)]
    best = {s:0.0}; parent={s:-1}; expanded=0

    while openq and expanded < max_expansions:
        f, g, u = heapq.heappop(openq); expanded += 1
        if u == t:
            path=[]; cur=u
            while cur!=-1: path.append(ids[cur]); cur=parent.get(cur,-1)
            path.reverse()
            return {"path":path,"cost":g,"expanded":expanded,"k":k,"exact":True}
        nn_idx, nn_d = _knn_of(u, X32, y, k)
        for v, wuv in zip(nn_idx, nn_d):
            g2 = g + float(wuv)
            if v not in best or g2 < best[v]-1e-12:
                best[v]=g2; parent[v]=u
                heapq.heappush(openq,(g2 + h(v), g2, v))
    return {"path":[],"cost":None,"expanded":expanded,"k":k,"exact":True,"note":"max expansions atteint"}

def _beam_k_path(ids, X, y, src_id: str, dst_id: str, k_neighbors: int = 10, K: int = 6, beam_width: int = 16):
    id2 = {ids[i]: i for i in range(len(ids))}
    if src_id not in id2 or dst_id not in id2:
        raise ValueError("A ou B introuvable dans le graph.")
    s, t = id2[src_id], id2[dst_id]
    X32 = X.astype(np.float32, copy=False)

    def h(i):
        diff = X32[i] - X32[t]
        return float(np.sqrt((diff*diff) @ y))

    beam=[(h(s),0.0,s,[ids[s]])]; best=None; expanded=0
    for _ in range(K):
        nxt=[]
        for score,g,u,path in beam:
            nn_idx, nn_d = _knn_of(u, X32, y, k_neighbors); expanded+=1
            for v, wuv in zip(nn_idx, nn_d):
                if ids[v] in path: continue
                g2=g+float(wuv); path2=path+[ids[v]]
                if v==t:
                    if best is None or g2<best[0]: best=(g2,path2)
                else:
                    nxt.append((g2 + h(v), g2, v, path2))
        if not nxt: break
        nxt.sort(key=lambda x:x[0]); beam=nxt[:beam_width]
        if best is not None:
            return {"path":best[1],"cost":best[0],"expanded":expanded,"k_neighbors":k_neighbors,"K":K,"beam":beam_width,"exact":False}
    if best is not None:
        return {"path":best[1],"cost":best[0],"expanded":expanded,"k_neighbors":k_neighbors,"K":K,"beam":beam_width,"exact":False}
    if beam:
        b=min(beam,key=lambda x:x[0])
        return {"path":b[3],"cost":b[1],"expanded":expanded,"k_neighbors":k_neighbors,"K":K,"beam":beam_width,"exact":False,"note":"but non atteint (meilleur partiel)"}
    return {"path":[],"cost":None,"expanded":expanded,"k_neighbors":k_neighbors,"K":K,"beam":beam_width,"exact":False,"note":"aucun chemin généré"}

@app.post("/path_astar")
def path_astar(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile | None = File(default=None),
    src: str = Query(...), dst: str = Query(...),
    y_vector: str | None = Query(default=None),
    k: int = Query(default=10, ge=2, le=200),
    maxexp: int = Query(default=20000, ge=100, le=1000000),
):
    """Chemin A* exact (bonus)."""
    try:
        ids, X = _load_graph(_read_csv(graph_file))
        qbytes = _read_csv(queries_file) if queries_file is not None else None
        if y_vector and y_vector.strip():
            y = _parse_vec50(y_vector, "Y_vector")
        else:
            y = np.ones(50, dtype=np.float32)
            if qbytes is not None:
                qdf = pd.read_csv(BytesIO(qbytes))
                if "Y_vector" in qdf.columns and len(qdf)>0:
                    y = _parse_vec50(qdf.iloc[0]["Y_vector"], "Y_vector")
        y = np.asarray(y, dtype=np.float32)
        t0=time.perf_counter()
        res=_astar_path(ids, X, y, src, dst, k=k, max_expansions=maxexp)
        res.update({"n_nodes":len(ids),"algo_time_sec":time.perf_counter()-t0})
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
    """Chemin Beam K-sauts (heuristique, bonus)."""
    try:
        ids, X = _load_graph(_read_csv(graph_file))
        qbytes = _read_csv(queries_file) if queries_file is not None else None
        if y_vector and y_vector.strip():
            y = _parse_vec50(y_vector, "Y_vector")
        else:
            y = np.ones(50, dtype=np.float32)
            if qbytes is not None:
                qdf = pd.read_csv(BytesIO(qbytes))
                if "Y_vector" in qdf.columns and len(qdf)>0:
                    y = _parse_vec50(qdf.iloc[0]["Y_vector"], "Y_vector")
        y = np.asarray(y, dtype=np.float32)
        t0=time.perf_counter()
        res=_beam_k_path(ids, X, y, src, dst, k_neighbors=k_neighbors, K=K, beam_width=beam)
        res.update({"n_nodes":len(ids),"algo_time_sec":time.perf_counter()-t0})
        return res
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ======================================================================
# ==============================  INTERFACE  ============================
# ======================================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html><html lang="fr"><head><meta charset="utf-8"/>
<title>TP Graph — Recherche & Chemins (complet)</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
:root{--bg:#f7f7f8;--card:#fff;--bd:#e6e6e7;--txt:#111}
*{box-sizing:border-box}body{font-family:system-ui,Inter,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--txt);margin:0}
.container{max-width:1100px;margin:32px auto;padding:0 16px}
h1{font-size:28px;margin:8px 0 20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:16px;margin:16px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:16px}
@media(max-width:860px){.grid2,.grid3{grid-template-columns:1fr}}
a.cta,button{padding:10px 16px;border-radius:10px;border:1px solid #111;background:#111;color:#fff;text-decoration:none;display:inline-block;cursor:pointer}
input,select{padding:8px;border:1px solid var(--bd);border-radius:10px}
label{display:block;font-size:13px;margin:10px 0 4px}
.small{font-size:13px;color:#555}
.badge{padding:3px 8px;border:1px solid var(--bd);border-radius:999px;font-size:12px}
.hidden{display:none}
.tabbar{display:flex;gap:8px;margin:12px 0 20px}
.tabbar button{background:transparent;color:#111;border:1px solid var(--bd)}
.tabbar button.active{background:#111;color:#fff;border-color:#111}
ul.small li{margin-bottom:6px;line-height:1.35}
code.key{background:#f0f0f0;padding:2px 6px;border-radius:6px;border:1px solid #e2e2e2}
</style></head>
<body><div class="container">
  <h1>Recherche pondérée & Chemins (Bonus)</h1>
  <div class="grid2">
    <div class="card">
      <h2>Recherche (naïf / pruned)</h2>
      <p class="small">Pour chaque requête (A, Y, D), retourne les nœuds B avec d(A,B) ≤ D.</p>
      <a href="#search" class="cta" onclick="show('search')">Ouvrir</a>
    </div>
    <div class="card">
      <h2>Chemins (A* / Beam)</h2>
      <p class="small">Minimisation de la somme des distances pondérées sur A→B (bonus).</p>
      <a href="#paths" class="cta" onclick="show('paths')">Ouvrir</a>
    </div>
  </div>

  <div id="search" class="card hidden">
    <div class="tabbar"><button id="tabS" class="active" onclick="show('search')">Recherche</button><button onclick="show('paths')">Chemins</button></div>
    <div class="grid3">
      <div>
        <h3>Rappels</h3>
        <ul class="small">
          <li><b>Naïf</b> : distance complète sur 50 features pour tous les nœuds.</li>
          <li><b>Pruned (Top-M)</b> : filtre rapide sur M features les plus pondérées (<b>borne</b>), puis distance complète <i>seulement</i> pour les survivants → même résultat que naïf.</li>
        </ul>
      </div>
      <div>
        <form onsubmit="runSearch(event)" class="card">
          <label>Graph CSV</label><input id="g1" type="file" accept=".csv" required>
          <label>Queries CSV</label><input id="q1" type="file" accept=".csv" required>
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px">
            <label>Mode</label>
            <select id="mode"><option value="naive">naïf</option><option value="pruned" selected>pruned</option></select>
            <label>Top-M</label><input id="topm" type="number" min="1" max="50" value="12" style="width:90px">
            <label>dscale</label><input id="dscale" type="number" step="0.05" min="0.05" max="2" value="1.0" style="width:90px">
          </div>
          <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
            <button type="button" onclick="runSummary()">Télécharger <b>résumé</b> (CSV)</button>
            <button type="button" onclick="runCompare()">Comparer (qui est plus rapide ?)</button>
          </div>
          <div id="sinfo" class="small" style="margin-top:8px"></div>
          <div id="cinfo" class="card small hidden"></div>
        </form>
      </div>
      <div>
        <h3>Paramètres (clairs)</h3>
        <ul class="small">
          <li><b>Mode</b> : <u>naïf</u> (référence), <u>pruned</u> (optimisé, identique).</li>
          <li><b>Top-M</b> : nombre de features utilisées pour la borne (12–16 conseillé).</li>
          <li><b>dscale</b> : applique D<sub>eff</sub>=D×dscale (<span class="small">pour démo, n’altère pas les CSV d’origine</span>).</li>
        </ul>
      </div>
    </div>
  </div>

  <div id="paths" class="card hidden">
    <div class="tabbar"><button onclick="show('search')">Recherche</button><button id="tabP" class="active" onclick="show('paths')">Chemins</button></div>
    <div class="grid3">
      <div>
        <h3>Comprendre</h3>
        <ul class="small">
          <li><b>A*</b> : exact (heuristique admissible = distance pondérée vers B).</li>
          <li><b>Beam</b> : heuristique (exploration limitée), plus rapide mais peut rater B.</li>
          <li>Voisinage k-NN recalculé à la volée selon <code>Y_vector</code>.</li>
        </ul>
      </div>
      <div>
        <form onsubmit="runPath(event)" class="card">
          <label>Graph CSV</label><input id="g2" type="file" accept=".csv" required>
          <label>Queries CSV (optionnel, pour Y par défaut)</label><input id="q2" type="file" accept=".csv">
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px">
            <label>A</label><input id="src" placeholder="node_1" style="width:120px">
            <label>B</label><input id="dst" placeholder="node_10" style="width:120px">
          </div>
          <label>Y_vector (50 valeurs ';' — vide: 1er Y des queries, sinon uniforme)</label>
          <input id="yvec" placeholder="y1;y2;...;y50" style="width:100%">
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:8px">
            <label>Algo</label>
            <select id="algo"><option value="astar" selected>A* (exact)</option><option value="beam">Beam (heuristique)</option></select>
            <span id="aParams" style="display:inline-flex;gap:8px;align-items:center">
              <label>k</label><input id="kA" type="number" min="2" max="200" value="10" style="width:90px">
              <label>MaxExp</label><input id="maxexp" type="number" min="100" max="1000000" value="20000" style="width:120px">
            </span>
            <span id="bParams" style="display:none;gap:8px;align-items:center">
              <label>k</label><input id="kB" type="number" min="2" max="200" value="10" style="width:90px">
              <label>K</label><input id="K" type="number" min="1" max="100" value="6" style="width:90px">
              <label>Beam</label><input id="BW" type="number" min="2" max="200" value="16" style="width:90px">
            </span>
          </div>
          <div style="margin-top:10px"><button type="submit">Calculer le chemin</button></div>
          <div id="pout" class="small" style="margin-top:8px"></div>
        </form>
      </div>
      <div>
        <h3>Conseils</h3>
        <ul class="small">
          <li>A* : augmentez <b>k</b> / <b>MaxExp</b> si A,B éloignés.</li>
          <li>Beam : augmentez <b>K</b> / <b>Beam</b> si “but non atteint”.</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<script>
function show(id){
  document.getElementById('search').classList.add('hidden');
  document.getElementById('paths').classList.add('hidden');
  document.getElementById(id).classList.remove('hidden');
  document.getElementById('tabS')?.classList.toggle('active', id==='search');
  document.getElementById('tabP')?.classList.toggle('active', id==='paths');
}
function onAlgo(){
  const v=document.getElementById('algo').value;
  document.getElementById('aParams').style.display=(v==='astar')?'inline-flex':'none';
  document.getElementById('bParams').style.display=(v==='beam')?'inline-flex':'none';
}
document.getElementById('algo').addEventListener('change', onAlgo); onAlgo();

async function runSummary(){
  const g=document.getElementById('g1').files[0];
  const q=document.getElementById('q1').files[0];
  const mode=document.getElementById('mode').value;
  const topm=parseInt(document.getElementById('topm').value||'12',10);
  const dscale=parseFloat(document.getElementById('dscale').value||'1');
  const sinfo=document.getElementById('sinfo'); sinfo.textContent='Résumé…';
  try{
    const fd=new FormData();
    fd.append('graph_file',g); fd.append('queries_file',q);
    fd.append('mode',mode); fd.append('top_m',String(topm)); fd.append('dscale',String(dscale));
    const res=await fetch('/search_summary_csv',{method:'POST',body:fd});
    const body=await res.blob();
    if(!res.ok){ sinfo.textContent=`Erreur: ${await body.text()}`; return; }
    const t=res.headers.get('X-Alg-Time');
    const url=URL.createObjectURL(body);
    const a=document.createElement('a'); a.href=url; a.download=`summary_${mode}.csv`; a.click(); URL.revokeObjectURL(url);
    sinfo.innerHTML=`<span class="badge">temps algo: ${Number(t).toFixed(3)}s</span>`;
  }catch(err){ sinfo.textContent='Erreur: '+(err?.message||err); }
}

async function runCompare(){
  const cinfo=document.getElementById('cinfo'); 
  cinfo.classList.remove('hidden'); 
  cinfo.textContent='Comparaison…';

  const g=document.getElementById('g1').files[0];
  const q=document.getElementById('q1').files[0];
  const topm=parseInt(document.getElementById('topm').value||'12',10);
  const dscale=parseFloat(document.getElementById('dscale').value||'1');

  try{
    const fd=new FormData(); 
    fd.append('graph_file',g); 
    fd.append('queries_file',q);
    fd.append('top_m',String(topm)); 
    fd.append('dscale',String(dscale));

    const res=await fetch('/compare_search',{method:'POST',body:fd});
    const j=await res.json();
    if(!res.ok || j.error){ 
      cinfo.textContent=`Erreur (${res.status}): ${j.error||''}`; 
      return; 
    }
    const faster = j.naive_time_sec < j.pruned_time_sec ? 'Naïf' : 'Pruned';
    cinfo.textContent = `Naïf: ${j.naive_time_sec.toFixed(3)}s | Pruned: ${j.pruned_time_sec.toFixed(3)}s → Plus rapide: ${faster}`;
  }catch(err){ 
    cinfo.textContent='Erreur: '+(err?.message||err); 
  }
}


async function runPath(e){
  e.preventDefault();
  const pout=document.getElementById('pout'); pout.textContent='Calcul…';
  const g=document.getElementById('g2').files[0]; const q=document.getElementById('q2').files[0];
  const src=document.getElementById('src').value.trim(); const dst=document.getElementById('dst').value.trim();
  const y=document.getElementById('yvec').value.trim(); const algo=document.getElementById('algo').value;
  try{
    const fd=new FormData(); fd.append('graph_file',g); if(q) fd.append('queries_file',q);
    let url='';
    if(algo==='astar'){
      const k=parseInt(document.getElementById('kA').value||'10',10);
      const me=parseInt(document.getElementById('maxexp').value||'20000',10);
      url=`/path_astar?src=${encodeURIComponent(src)}&dst=${encodeURIComponent(dst)}&k=${k}&maxexp=${me}`;
    }else{
      const k2=parseInt(document.getElementById('kB').value||'10',10);
      const K=parseInt(document.getElementById('K').value||'6',10);
      const BW=parseInt(document.getElementById('BW').value||'16',10);
      url=`/path_beam?src=${encodeURIComponent(src)}&dst=${encodeURIComponent(dst)}&k_neighbors=${k2}&K=${K}&beam=${BW}`;
    }
    if(y) url += `&y_vector=${encodeURIComponent(y)}`;
    const res=await fetch(url,{method:'POST',body:fd});
    const raw=await res.text(); let j; try{ j=JSON.parse(raw);} catch{ j={error:'Réponse non-JSON', body:raw}; }
    if(!res.ok || j.error){ pout.textContent=`Erreur (${res.status}): ${j.error||j.body}`; return; }
    const path=(j.path||[]).join(' → '); const exact=j.exact?'A* (exact)':'Beam (heuristique)'; const note=j.note?(' | '+j.note):'';
    pout.textContent=`Algo: ${exact} | nœuds: ${j.n_nodes} | expansions: ${j.expanded}
Coût total: ${j.cost!=null?j.cost.toFixed(6):'-'} | Temps algo: ${j.algo_time_sec?j.algo_time_sec.toFixed(3)+'s':'-'}
Chemin: ${path||'-'}${note}`;
  }catch(err){ pout.textContent='Erreur: '+(err?.message||err); }
}
</script>
</body></html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))
