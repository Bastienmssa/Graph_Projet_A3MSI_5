# api.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn, os, re, heapq, time
import numpy as np
import pandas as pd
from io import BytesIO, StringIO

app = FastAPI(title="TP Graph — Recherche & Bonus (fix IDs)", version="7.1")

# ========= Chargement & colonnes =========

def _read_csv_flex(up: UploadFile) -> bytes:
    # CSV only (le sujet); on lit brut
    return up.file.read()

def _detect_id_col(df: pd.DataFrame) -> str:
    for c in ["node_id","id","ID","Id","node","Node","name","Name","point_A","A"]:
        if c in df.columns: return c
    for c in df.columns:
        if df[c].dtype == object: return c
    return df.columns[0]

def _load_graph_from_csv_bytes(csv_bytes: bytes):
    df = pd.read_csv(BytesIO(csv_bytes)).dropna(axis=1, how="all")
    id_col = _detect_id_col(df)

    # Cas 1: Y_vector déjà présent
    if "Y_vector" in df.columns:
        ids = df[id_col].astype(str).values
        X = df["Y_vector"].astype(str).apply(lambda s:[float(x) for x in str(s).split(";") if x!=""]).to_list()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 50:
            raise ValueError("Le graphe doit avoir 50 valeurs par nœud (50 dims).")
        return ids, X

    # Cas 2: features colonnes numériques (on en garde 50)
    feat_named = [c for c in df.columns if re.match(r"(?i)feature[_-]?\d+$", str(c))]
    if len(feat_named) >= 50:
        cols = sorted(feat_named, key=lambda x:int(re.findall(r"\d+", x)[0]))[:50]
        feats = df[cols]
    else:
        feats = df.drop(columns=list({id_col,"D","point_A","A","Y_vector"} & set(df.columns)))
        feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if feats.shape[1] > 50:
            variances = feats.var(axis=0, ddof=0)
            feats = feats[variances.sort_values(ascending=False).index[:50]]
    if feats.shape[1] != 50:
        raise ValueError(f"Après sélection: {feats.shape[1]} colonnes features (50 attendues).")
    ids = df[id_col].astype(str).values
    X = feats.to_numpy(dtype=np.float32)
    return ids, X

def _parse_y(s: str) -> np.ndarray:
    arr = np.asarray([float(x) for x in str(s).split(";") if x!=""], dtype=np.float32)
    if arr.shape != (50,): raise ValueError("Y_vector doit contenir exactement 50 valeurs.")
    return arr

def _default_y_from_queries_csv(qfile_bytes: bytes | None) -> np.ndarray:
    if not qfile_bytes: return np.ones(50, dtype=np.float32)
    try:
        qdf = pd.read_csv(BytesIO(qfile_bytes))
        if "Y_vector" in qdf.columns and len(qdf) > 0:
            y = _parse_y(str(qdf.iloc[0]["Y_vector"]))
            return np.asarray(y, dtype=np.float32)
    except Exception:
        pass
    return np.ones(50, dtype=np.float32)

def _extract_queries_cols(qdf: pd.DataFrame):
    colA = "point_A" if "point_A" in qdf.columns else ("A" if "A" in qdf.columns else None)
    if colA is None: raise ValueError("Queries: colonne 'point_A' (ou 'A') absente.")
    if "D" not in qdf.columns: raise ValueError("Queries: colonne 'D' absente.")
    return colA

# ========= Normalisation d’IDs (clé du fix) =========

_num_pat = re.compile(r"(\d+)$")
_prefix_pat = re.compile(r"^(\D*?)(\d+)$")

def _infer_graph_prefix(ids: np.ndarray) -> str:
    # déduit le préfixe non-numérique du 1er id si possible (ex: 'node_' pour 'node_123')
    s = str(ids[0])
    m = _prefix_pat.match(s)
    return m.group(1) if m else ""

def _normalize_A(A: str, id_to_idx: dict, graph_prefix: str) -> str | None:
    # si déjà présent → OK
    if A in id_to_idx: return A
    # sinon, on prend la partie numérique finale et on reconstruit prefix+digits
    m = _num_pat.search(A)
    if not m: return None
    cand = f"{graph_prefix}{m.group(1)}"
    if cand in id_to_idx: return cand
    return None

# ========= Distances & kNN =========

def _w2_dist2(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = X - xA
    return (diff*diff) @ y

def _knn_of(idx: int, X: np.ndarray, y: np.ndarray, k: int):
    xA = X[idx]
    d2 = _w2_dist2(X, xA, y)
    d2[idx] = np.inf
    k = min(k, X.shape[0]-1)
    nn_idx = np.argpartition(d2, k)[:k]
    order = np.argsort(d2[nn_idx], kind="stable")
    nn_idx = nn_idx[order]
    return nn_idx, np.sqrt(d2[nn_idx])

# ========= Recherche (naïf / pruned avec normalisation d’IDs) =========

def _search_naive(ids, X, qdf, colA, dscale: float):
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    graph_prefix = _infer_graph_prefix(ids)
    out_rows = []; matched=0; skipped=0
    t0 = time.perf_counter()
    for qi, row in qdf.iterrows():
        Araw = str(row[colA])
        A = _normalize_A(Araw, id_to_idx, graph_prefix)
        if A is None:
            skipped += 1
            continue
        matched += 1
        D = float(row["D"]); Deff2 = float((D*dscale)**2)
        if "Y_vector" in qdf.columns and isinstance(row.get("Y_vector", None), str) and row["Y_vector"].strip():
            y = _parse_y(str(row["Y_vector"]))
        else:
            y = _default_y_from_queries_csv(qdf.to_csv(index=False).encode())
        idxA = id_to_idx[A]
        d2 = _w2_dist2(X, X[idxA], y); d2[idxA]=np.inf
        sel = np.where(d2 <= Deff2)[0]
        if sel.size:
            d = np.sqrt(d2[sel]); ord_ = np.argsort(d, kind="stable")
            for j, dist in zip(sel[ord_], d[ord_]):
                out_rows.append((qi, Araw, ids[j], float(dist)))
    return out_rows, time.perf_counter()-t0, matched, skipped

def _search_pruned(ids, X, qdf, colA, dscale: float, top_m: int):
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    graph_prefix = _infer_graph_prefix(ids)
    out_rows = []; pruned_rejects=0; matched=0; skipped=0
    t0 = time.perf_counter()
    for qi, row in qdf.iterrows():
        Araw = str(row[colA])
        A = _normalize_A(Araw, id_to_idx, graph_prefix)
        if A is None:
            skipped += 1
            continue
        matched += 1
        D = float(row["D"]); Deff2 = float((D*dscale)**2)
        if "Y_vector" in qdf.columns and isinstance(row.get("Y_vector", None), str) and row["Y_vector"].strip():
            y = _parse_y(str(row["Y_vector"]))
        else:
            y = _default_y_from_queries_csv(qdf.to_csv(index=False).encode())
        m = max(1, min(int(top_m), 50))
        top_idx = np.argsort(-y)[:m]
        XA = X[id_to_idx[A]]
        diffM = X[:, top_idx] - XA[top_idx]
        lb2 = (diffM*diffM) @ y[top_idx]; lb2[id_to_idx[A]] = np.inf
        cand = np.where(lb2 <= Deff2)[0]; pruned_rejects += int((lb2 > Deff2).sum())
        if cand.size:
            d2 = _w2_dist2(X[cand], XA, y)
            mask = d2 <= Deff2
            cand2 = cand[mask]
            if cand2.size:
                d = np.sqrt(d2[mask]); ord_ = np.argsort(d, kind="stable")
                for j, dist in zip(cand2[ord_], d[ord_]):
                    out_rows.append((qi, Araw, ids[j], float(dist)))
    return out_rows, time.perf_counter()-t0, pruned_rejects, matched, skipped

@app.post("/search_csv")
def search_csv(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    mode: str = Query(default="pruned"),              # "naive" | "pruned"
    top_m: int = Query(default=12, ge=1, le=50),
    dscale: float = Query(default=1.0, ge=0.05, le=2.0),
):
    try:
        gbytes = _read_csv_flex(graph_file)
        qbytes = _read_csv_flex(queries_file)
        ids, X = _load_graph_from_csv_bytes(gbytes)
        qdf = pd.read_csv(BytesIO(qbytes))
        colA = _extract_queries_cols(qdf)

        if mode == "naive":
            rows, t, matched, skipped = _search_naive(ids, X, qdf, colA, dscale)
            pruned_rejects = 0
        elif mode == "pruned":
            rows, t, pruned_rejects, matched, skipped = _search_pruned(ids, X, qdf, colA, dscale, top_m)
        else:
            raise ValueError("mode doit être 'naive' ou 'pruned'.")

        # CSV de sortie
        buf = StringIO()
        buf.write("query_idx,point_A,node,dist\n")
        for qi, A, nid, dist in rows:
            buf.write(f"{qi},{A},{nid},{dist:.6f}\n")

        headers = {
            "X-Alg-Time": f"{t:.6f}",
            "X-Row-Count": str(len(rows)),
            "X-PrunedRejects": str(pruned_rejects),
            "X-Matched": str(matched),
            "X-Skipped": str(skipped),
            "Content-Disposition": f'attachment; filename="{mode}_results.csv"',
        }
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/compare_search")
def compare_search(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    top_m: int = Query(default=12, ge=1, le=50),
    dscale: float = Query(default=1.0, ge=0.05, le=2.0),
):
    try:
        gbytes = _read_csv_flex(graph_file)
        qbytes = _read_csv_flex(queries_file)
        ids, X = _load_graph_from_csv_bytes(gbytes)
        qdf = pd.read_csv(BytesIO(qbytes))
        colA = _extract_queries_cols(qdf)

        N, tN, mN, sN = _search_naive(ids, X, qdf, colA, dscale)
        P, tP, rej, mP, sP = _search_pruned(ids, X, qdf, colA, dscale, top_m)
        equal = (sorted(N) == sorted(P))

        return {
            "naive_time_sec": tN,
            "pruned_time_sec": tP,
            "rows_naive": len(N),
            "rows_pruned": len(P),
            "equal": equal,
            "pruned_rejects": rej,
            "matched_queries": mN,
            "skipped_queries": sN
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ========= A* exact & Beam (repris de ta version qui marche) =========

def _astar_path(ids, X, y, src_id: str, dst_id: str, k: int = 10, max_expansions: int = 20000):
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

# ========= Endpoints Bonus =========

@app.post("/path_astar")
def path_astar(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile | None = File(default=None),
    src: str = Query(...), dst: str = Query(...),
    y_vector: str | None = Query(default=None),
    k: int = Query(default=10, ge=2, le=200),
    maxexp: int = Query(default=20000, ge=100, le=1000000),
):
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

# ========= UI (2 écrans) =========

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html><html lang="fr"><head><meta charset="utf-8"/>
<title>TP Graph — Recherche & Bonus</title>
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
</style></head>
<body><div class="container">
  <h1>Recherche pondérée & Chemins (Bonus)</h1>
  <div class="grid2">
    <div class="card">
      <h2>Recherche (naïf / pruned)</h2>
      <p class="small">Pour chaque (A,Y,D), retourne les B avec d(A,B) ≤ D.</p>
      <a href="#search" class="cta" onclick="show('search')">Ouvrir</a>
    </div>
    <div class="card">
      <h2>Chemins (A* / Beam)</h2>
      <p class="small">Minimise la somme des distances pondérées le long de A→B.</p>
      <a href="#paths" class="cta" onclick="show('paths')">Ouvrir</a>
    </div>
  </div>

  <div id="search" class="card hidden">
    <div class="tabbar"><button id="tabS" class="active" onclick="show('search')">Recherche</button><button onclick="show('paths')">Chemins</button></div>
    <div class="grid3">
      <div>
        <h3>Comprendre</h3>
        <ul class="small">
          <li><b>Naïf</b> : distance complète (50 dims).</li>
          <li><b>Pruned</b> : borne sur Top-M poids de Y, puis calcul complet (résultat = naïf).</li>
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
            <button type="submit">Télécharger résultats (CSV)</button>
            <button type="button" onclick="runCompare()">Comparer naïf vs pruned</button>
          </div>
          <div id="sinfo" class="small" style="margin-top:8px"></div>
          <div id="cinfo" class="card small hidden"></div>
        </form>
      </div>
      <div>
        <h3>Paramètres</h3>
        <ul class="small"><li><b>Top-M</b> : nb de features pour la borne (def. 12).</li><li><b>dscale</b> : D<sub>eff</sub>=D×dscale (démo, non destructif).</li></ul>
      </div>
    </div>
  </div>

  <div id="paths" class="card hidden">
    <div class="tabbar"><button onclick="show('search')">Recherche</button><button id="tabP" class="active" onclick="show('paths')">Chemins</button></div>
    <div class="grid3">
      <div>
        <h3>Comprendre</h3>
        <ul class="small">
          <li>Graphe k-NN (selon Y). <b>A*</b> = exact ; <b>Beam</b> = heuristique (K, Beam).</li>
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
        <ul class="small"><li>A* : augmentez k / MaxExp si A,B éloignés.</li><li>Beam : augmentez K / Beam si but non atteint.</li></ul>
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

async function runSearch(e){
  e.preventDefault();
  const g=document.getElementById('g1').files[0];
  const q=document.getElementById('q1').files[0];
  const mode=document.getElementById('mode').value;
  const topm=parseInt(document.getElementById('topm').value||'12',10);
  const dscale=parseFloat(document.getElementById('dscale').value||'1');
  const sinfo=document.getElementById('sinfo'); sinfo.textContent='Calcul…';
  try{
    const fd=new FormData();
    fd.append('graph_file',g); fd.append('queries_file',q);
    fd.append('mode',mode); fd.append('top_m',String(topm)); fd.append('dscale',String(dscale));
    const res=await fetch('/search_csv',{method:'POST',body:fd});
    const body=await res.blob();
    if(!res.ok){ const txt=await body.text(); sinfo.textContent=`Erreur (${res.status}): ${txt}`; return; }
    const t=res.headers.get('X-Alg-Time'), n=res.headers.get('X-Row-Count');
    const pr=res.headers.get('X-PrunedRejects'), m=res.headers.get('X-Matched'), s=res.headers.get('X-Skipped');
    const url=URL.createObjectURL(body);
    const a=document.createElement('a'); a.href=url; a.download=`${mode}_results.csv`; a.click(); URL.revokeObjectURL(url);
    sinfo.innerHTML=`<span class="badge">algo: ${Number(t).toFixed(3)}s</span> <span class="badge">lignes: ${n}</span> <span class="badge">queries OK: ${m}</span> <span class="badge">ignorées: ${s}</span> ${mode==='pruned'?`<span class="badge">pruned rejects: ${pr}</span>`:''}`;
  }catch(err){ sinfo.textContent='Erreur: '+(err?.message||err); }
}

async function runCompare(){
  const cinfo=document.getElementById('cinfo'); cinfo.classList.remove('hidden'); cinfo.textContent='Comparaison…';
  const g=document.getElementById('g1').files[0]; const q=document.getElementById('q1').files[0];
  const topm=parseInt(document.getElementById('topm').value||'12',10);
  const dscale=parseFloat(document.getElementById('dscale').value||'1');
  try{
    const fd=new FormData();
    fd.append('graph_file',g); fd.append('queries_file',q);
    fd.append('top_m',String(topm)); fd.append('dscale',String(dscale));
    const res=await fetch('/compare_search',{method:'POST',body:fd});
    const j=await res.json();
    if(!res.ok || j.error){ cinfo.textContent=`Erreur (${res.status}): ${j.error}`; return; }
    const eq=j.equal?'✅ identiques':'❌ différents';
    cinfo.textContent=`naïf: ${j.rows_naive} lignes, ${j.naive_time_sec.toFixed(3)}s
pruned: ${j.rows_pruned} lignes, ${j.pruned_time_sec.toFixed(3)}s
pruned rejects: ${j.pruned_rejects}
queries OK: ${j.matched_queries} | ignorées: ${j.skipped_queries}
égalité: ${eq}`;
  }catch(err){ cinfo.textContent='Erreur: '+(err?.message||err); }
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
    const raw=await res.text();
    let j; try{ j=JSON.parse(raw);} catch{ j={error:'Réponse non-JSON', body:raw}; }
    if(!res.ok || j.error){ pout.textContent=`Erreur (${res.status}): ${j.error||j.body}`; return; }
    const path=(j.path||[]).join(' → ');
    const exact=j.exact?'A* (exact)':'Beam (heuristique)';
    const note=j.note?(' | '+j.note):'';
    pout.textContent=`Algo: ${exact} | nœuds: ${j.n_nodes} | expansions: ${j.expanded}
Coût total: ${j.cost!=null?j.cost.toFixed(6):'-'} | Temps algo: ${j.algo_time_sec?j.algo_time_sec.toFixed(3)+'s':'-'}
Chemin: ${path||'-'}${note}`;
  }catch(err){ pout.textContent='Erreur: '+(err?.message||err); }
}
</script>
</body></html>
"""

if __name__ == "__main__":
    np.random.seed(0)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))
