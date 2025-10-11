from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uvicorn
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
import re, time

# Mode "fast" = PCA + KDTree pour candidatage + rescorage EXACT
try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    SK_OK = True
except Exception:
    SK_OK = False

app = FastAPI(title="Recherche pondérée sur graphe", version="1.1")

# ---------------------------------------------------
#                UTILITAIRES DE CHARGEMENT
# ---------------------------------------------------
def _detect_id_col(df: pd.DataFrame) -> str:
    for c in ["node_id","id","ID","Id","node","Node","name","Name","point_A","A"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]

def _load_graph_from_file(fobj):
    df = pd.read_csv(fobj)
    df = df.dropna(axis=1, how="all")
    id_col = _detect_id_col(df)

    # 50 features : soit colonne Y_vector, soit feature_1..50, soit sélection par variance
    if "Y_vector" in df.columns:
        ids = df[id_col].astype(str).values
        X = df["Y_vector"].astype(str).apply(lambda s: [float(x) for x in str(s).split(";") if x!=""]).to_list()
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != 50:
            raise ValueError(f"Le graphe doit contenir 50 valeurs par nœud (trouvé {X.shape[1]}).")
        return ids, X

    feat_named = [c for c in df.columns if re.match(r"(?i)feature[_-]?\d+$", str(c))]
    if len(feat_named) >= 50:
        feat_named_sorted = sorted(feat_named, key=lambda x: int(re.findall(r"\d+", x)[0]))[:50]
        feats = df[feat_named_sorted]
    else:
        drop_cols = {id_col,"D","point_A","A"} & set(df.columns)
        feats = df.drop(columns=list(drop_cols))
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
    qdf = pd.read_csv(fobj)
    need = {"point_A","Y_vector","D"}
    if not need.issubset(set(qdf.columns)):
        raise ValueError("Le fichier queries doit contenir : point_A, Y_vector, D")
    qdf["point_A"] = qdf["point_A"].astype(str)
    qdf["D"] = pd.to_numeric(qdf["D"], errors="coerce").astype(float)
    if qdf["D"].isna().any():
        raise ValueError("Colonne D invalide.")
    return qdf

def _graph_prefix(ids: np.ndarray) -> str:
    s = str(ids[0])
    return s.split("_",1)[0] if "_" in s else "node"

def _map_pointA_to_graph_prefix(qdf: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Mappe automatiquement ads_1 -> node_1 si besoin, sans toucher aux CSV."""
    if qdf["point_A"].isin(ids).all():
        return qdf
    prefix = _graph_prefix(ids)
    q = qdf.copy()
    q["point_A"] = q["point_A"].astype(str).apply(lambda s: re.sub(r"^[A-Za-z]+_", f"{prefix}_", s))
    return q

def _parse_y(s: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(s).split(";") if x!=""], dtype=np.float32)

# ---------------------------------------------------
#                DISTANCE & RECHERCHE
# ---------------------------------------------------
def _w2_dist(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Somme(y_i * (Xi - xA_i)^2) pour tous les nœuds (retourne dist^2)."""
    diff = X - xA
    return (diff*diff) @ y

def _search_naive(ids: np.ndarray, X: np.ndarray, qdf: pd.DataFrame, topk: int|None):
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    X32 = X.astype(np.float32, copy=False)
    out = []
    for qi, row in qdf.iterrows():
        a = row["point_A"]
        if a not in id_to_idx:
            continue
        y = _parse_y(row["Y_vector"])
        D = float(row["D"])
        aidx = id_to_idx[a]
        d2 = _w2_dist(X32, X32[aidx], y)
        mask = d2 <= (D*D)
        idxs = np.where(mask)[0]
        if topk is not None and topk < len(idxs):
            part = np.argpartition(d2[idxs], topk-1)[:topk]
            idxs = idxs[part]
        order = np.argsort(d2[idxs], kind="stable")
        idxs = idxs[order]
        if len(idxs)==0:
            continue
        out.append(pd.DataFrame({
            "query_idx": qi,
            "point_A": a,
            "node": ids[idxs],
            "dist": np.sqrt(d2[idxs]).astype(np.float32)
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["query_idx","point_A","node","dist"])

def _search_fast(ids: np.ndarray, X: np.ndarray, qdf: pd.DataFrame, topk: int|None):
    """
    Amélioration (étape 4) :
    - PCA(50→15) + KDTree pour générer un pool de CANDIDATS (euclidien, espace réduit),
    - puis RESCORAGE EXACT en 50D avec la distance pondérée (mêmes 'dist' que naïve),
    - pool ADAPTATIF : si 0 résultat ou pas assez (<TopK), on agrandit le pool jusqu'à N.
    """
    if not SK_OK:
        raise RuntimeError("Le mode 'fast' nécessite scikit-learn (pip install scikit-learn).")

    X32 = X.astype(np.float32, copy=False)
    pca = PCA(n_components=15, random_state=0)
    X_red = pca.fit_transform(X32)
    nbrs = NearestNeighbors(algorithm="kd_tree").fit(X_red)

    id_to_idx = {ids[i]: i for i in range(len(ids))}
    out = []

    N = len(ids)
    BASE_K = 256 if N <= 2000 else 512
    MAX_K = N

    for qi, row in qdf.iterrows():
        a = row["point_A"]
        if a not in id_to_idx:
            continue
        D = float(row["D"])
        y = _parse_y(row["Y_vector"])
        aidx = id_to_idx[a]

        cand_k = min(BASE_K, MAX_K)
        best_nodes = None
        best_dist = None

        while True:
            # 1) candidats rapides (PCA + KDTree)
            _, idxs = nbrs.kneighbors([X_red[aidx]], n_neighbors=cand_k, return_distance=True)
            cand = idxs[0]
            if cand.size == 0:
                break

            # 2) rescorage EXACT en 50D
            xA = X32[aidx]
            diff = X32[cand] - xA
            d2 = (diff * diff) @ y
            dist = np.sqrt(d2).astype(np.float32)

            # 3) filtre rayon + tri
            mask = dist <= D
            if np.any(mask):
                take_idx = np.where(mask)[0]
                order = np.argsort(dist[take_idx], kind="stable")
                take_idx = take_idx[order]
                nodes = cand[take_idx]
                dists = dist[take_idx]

                if topk is not None and len(nodes) > topk:
                    nodes = nodes[:topk]
                    dists = dists[:topk]

                best_nodes = nodes
                best_dist = dists

            # stop si on a assez (>= topk) ou si on a exploré tout le graphe
            if (best_nodes is not None and (topk is None or len(best_nodes) >= topk)) or cand_k >= MAX_K:
                break

            # sinon on agrandit le pool
            cand_k = min(MAX_K, cand_k * 2)

        if best_nodes is None or len(best_nodes) == 0:
            continue

        out.append(pd.DataFrame({
            "query_idx": qi,
            "point_A": a,
            "node": ids[best_nodes],
            "dist": best_dist
        }))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["query_idx","point_A","node","dist"])

# ---------------------------------------------------
#                ENDPOINT + STATS SERVEUR
# ---------------------------------------------------
@app.post("/search")
def search(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    mode: str = Query(default="naive", pattern="^(naive|fast)$"),
    topk: int | None = Query(default=None),
    fmt: str = Query(default="json", pattern="^(json|csv)$"),
):
    t0 = time.time()
    try:
        gid, qid = BytesIO(graph_file.file.read()), BytesIO(queries_file.file.read())
        ids, X = _load_graph_from_file(gid)
        qid.seek(0)
        qdf = _load_queries_from_file(qid)
        qdf = _map_pointA_to_graph_prefix(qdf, ids)

        if len(ids) < 500 or len(ids) > 5000:
            return JSONResponse(status_code=400, content={"error":"Le graphe doit contenir entre 500 et 5000 nœuds."})

        t1 = time.time()
        if mode == "naive":
            res = _search_naive(ids, X, qdf, topk)
        else:
            res = _search_fast(ids, X, qdf, topk)
        t2 = time.time()

        stats = {
            "mode": mode,
            "n_nodes": len(ids),
            "n_queries": len(qdf),
            "total_time_sec": round(t2 - t0, 4),
            "algo_time_sec": round(t2 - t1, 4),
            "avg_per_query_sec": round((t2 - t1) / max(1, len(qdf)), 6),
        }

        if fmt == "json":
            return JSONResponse({"results": res.to_dict(orient="records"), "stats": stats})
        else:
            buf = StringIO()
            res.to_csv(buf, index=False)
            buf.seek(0)
            return StreamingResponse(
                iter([buf.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition":"attachment; filename=results_all.csv"}
            )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ---------------------------------------------------
#                INTERFACE WEB (TIMER CLIENT)
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!doctype html>
    <html lang=fr>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Recherche pondérée sur graphe</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px}
        header{margin-bottom:24px}
        .card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0}
        input[type=file]{display:block;margin:8px 0}
        button{padding:10px 16px;border-radius:10px;border:1px solid #111;background:#111;color:#fff;cursor:pointer}
        table{border-collapse:collapse;width:100%}
        th,td{border:1px solid #e5e7eb;padding:8px;text-align:left}
        th{background:#f9fafb}
        .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
      </style>
    </head>
    <body>
      <header>
        <h1>Recherche pondérée sur graphe</h1>
        <p>Chargez <b>Graph CSV</b> + <b>Queries CSV</b>, choisissez le <b>mode</b>, puis lancez.</p>
      </header>
      <section class="card">
        <form id="f" onsubmit="run(event)">
          <label>Graph CSV</label>
          <input id="graph" type="file" accept=".csv" required />
          <label>Queries CSV</label>
          <input id="queries" type="file" accept=".csv" required />
          <div class="row">
            <label>Mode</label>
            <select id="mode"><option>naive</option><option>fast</option></select>
            <label>Top-K (optionnel)</label>
            <input id="topk" type="number" min="1" step="1" style="width:120px" />
            <label>Format</label>
            <select id="fmt"><option>json</option><option>csv</option></select>
            <button type="submit">Lancer la recherche</button>
          </div>
        </form>
      </section>
      <section id="out" class="card" style="display:none"></section>
      <script>
      async function run(e){
        e.preventDefault();
        const t0 = performance.now(); // timer client: au clic

        const graph = document.getElementById('graph').files[0];
        const queries = document.getElementById('queries').files[0];
        const mode = document.getElementById('mode').value;
        const topk = document.getElementById('topk').value;
        const fmt = document.getElementById('fmt').value;

        const fd = new FormData();
        fd.append('graph_file', graph);
        fd.append('queries_file', queries);

        const url = '/search?mode='+mode + (topk?('&topk='+topk):'') + '&fmt='+fmt;

        const out = document.getElementById('out');
        out.style.display='block';
        out.innerHTML = '<em>Calcul en cours...</em>';

        const res = await fetch(url,{method:'POST',body:fd});

        if(!res.ok){
          const j = await res.json().catch(()=>({error:'Erreur inconnue'}));
          out.innerHTML = `<h3>Erreur</h3><pre>${(j.error||JSON.stringify(j))}</pre>`;
          return;
        }

        if(fmt==='csv'){
          const blob = await res.blob();
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = 'results_all.csv';
          const t1 = performance.now();
          const clientSec = ((t1 - t0)/1000).toFixed(4);
          out.innerHTML = `<h3>Résultats</h3>
            <p><a href="${a.href}" download="results_all.csv">Télécharger results_all.csv</a></p>
            <pre>Client elapsed (clic → prêt): ${clientSec}s</pre>`;
          return;
        }

        const j = await res.json();
        const rows = j.results || [];
        const stats = j.stats || {};
        if(rows.length===0){
          const t1 = performance.now();
          const clientSec = ((t1 - t0)/1000).toFixed(4);
          out.innerHTML = `<p>Aucun résultat.</p>
            <pre>Client elapsed (clic → affichage): ${clientSec}s</pre>`;
          return;
        }

        const head = Object.keys(rows[0]);
        let html = '<h3>Résultats</h3><div style="max-height:420px;overflow:auto"><table><thead><tr>' + head.map(h=>`<th>${h}</th>`).join('') + '</tr></thead><tbody>';
        for(const r of rows){
          html += '<tr>' + head.map(h=>`<td>${r[h]}</td>`).join('') + '</tr>';
        }
        html += '</tbody></table></div>';

        html += `<pre style="margin-top:12px;background:#f9fafb;padding:10px;border-radius:10px">
Stats (serveur):
- Mode: ${stats.mode}
- Nœuds: ${stats.n_nodes}
- Requêtes: ${stats.n_queries}
- Temps total (serveur): ${stats.total_time_sec}s
- Temps algo (serveur): ${stats.algo_time_sec}s
- Moyenne par requête: ${stats.avg_per_query_sec}s
</pre>`;

        const t1 = performance.now(); // timer client: affichage prêt
        const clientSec = ((t1 - t0)/1000).toFixed(4);
        html += `<pre>Client elapsed (clic → affichage): ${clientSec}s</pre>`;

        out.innerHTML = html;
      }
      </script>
    </body>
    </html>
    """

# ---------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
