from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
import re, time

app = FastAPI(title="Recherche pondérée — CSV + Comparaison (naive vs pruned Top-M)", version="3.0")

# ---------- Chargement & utilitaires ----------

# Détecte la colonne d'ID dans le CSV (node_id, id, node, etc.) pour être robuste aux noms.
def _detect_id_col(df: pd.DataFrame) -> str:
    for c in ["node_id","id","ID","Id","node","Node","name","Name","point_A","A"]:
        if c in df.columns: return c
    for c in df.columns:
        if df[c].dtype == object: return c
    return df.columns[0]

# Charge le CSV "graph" et renvoie (ids, X):
#  - ids : identifiants des nœuds
#  - X   : matrice (N,50) des 50 features numériques par nœud
def _load_graph_from_file(fobj):
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

# Charge le CSV "queries" et vérifie: point_A, Y_vector, D.
# point_A = id de A ; Y_vector = "y1;...;y50" ; D = rayon.
def _load_queries_from_file(fobj):
    qdf = pd.read_csv(fobj)
    need = {"point_A","Y_vector","D"}
    if not need.issubset(set(qdf.columns)):
        raise ValueError("Le fichier queries doit contenir : point_A, Y_vector, D")
    qdf["point_A"] = qdf["point_A"].astype(str)
    qdf["D"] = pd.to_numeric(qdf["D"], errors="coerce").astype(float)
    if qdf["D"].isna().any(): raise ValueError("Colonne D invalide.")
    return qdf

# Aligne le préfixe des ids si nécessaire (ex: ads_1 -> node_1) pour éviter les mismatches.
def _graph_prefix(ids: np.ndarray) -> str:
    s = str(ids[0]);  return s.split("_",1)[0] if "_" in s else "node"

def _map_pointA_to_graph_prefix(qdf: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    if qdf["point_A"].isin(ids).all(): return qdf
    prefix = _graph_prefix(ids)
    q = qdf.copy()
    q["point_A"] = q["point_A"].astype(str).apply(
        lambda s: re.sub(r"^[A-Za-z]+_", f"{prefix}_", s)
    )
    return q

# Parse "y1;...;y50" en array(50,). Erreur si la taille != 50 (sécurité).
def _parse_y(s: str) -> np.ndarray:
    arr = np.asarray([float(x) for x in str(s).split(";") if x!=""], dtype=np.float32)
    if arr.size != 50:
        raise ValueError(f"Y_vector doit contenir 50 coefficients (reçu {arr.size}).")
    return arr

# ---------- Distance pondérée & LB (pruning) ----------

# Calcule d^2(A,B) pondérée pour tous les B : (X - xA)^2 @ y (vectorisé NumPy).
def _w2_dist2(X: np.ndarray, xA: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = X - xA
    return (diff*diff) @ y

# Indices des M plus grands poids de y (Top-M(Y)) pour le test rapide.
def _topM_idx(y: np.ndarray, M: int) -> np.ndarray:
    M = max(1, min(M, y.shape[0]))
    return np.argpartition(-y, M-1)[:M]

# Borne inférieure LB^2 en sommant seulement sur les indices Top-M.
# LB^2 <= d^2 vraie -> si LB^2 > D^2, rejet garanti sans calculer le reste.
def _partial_lb_dist2(X: np.ndarray, xA: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    diff = X[:, idx] - xA[idx]
    return (diff*diff) @ y[idx]

# ---------- Deux modes : naive & pruned (Top-M) ----------

# Mode "naive" : brute force exact (50 features), garde d <= D_eff.
def _search_naive(ids, X, qdf, dmult):
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    X32 = X.astype(np.float32, copy=False)
    out = []
    EPS = 1e-12
    for qi, row in qdf.iterrows():
        a = row["point_A"];  D = float(row["D"]) * float(dmult)
        if a not in id_to_idx: continue
        y = _parse_y(row["Y_vector"])
        aidx = id_to_idx[a]
        d2 = _w2_dist2(X32, X32[aidx], y)
        mask = d2 <= (D*D + EPS)
        idxs = np.where(mask)[0]
        if idxs.size == 0: continue
        order = np.argsort(d2[idxs], kind="stable")
        idxs = idxs[order]
        out.append(pd.DataFrame({
            "query_idx": qi, "point_A": a,
            "node": ids[idxs], "dist": np.sqrt(d2[idxs]).astype(np.float32)
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["query_idx","point_A","node","dist"])

# Mode "pruned" : exact + test rapide Top-M (sélectionnable), puis distance 50D pour les survivants.
def _search_pruned_topM(ids, X, qdf, dmult, M=12):
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    X32 = X.astype(np.float32, copy=False)
    out = []
    pruned_total = 0
    tested_total = 0
    EPS = 1e-12

    for qi, row in qdf.iterrows():
        a = row["point_A"];  D = float(row["D"]) * float(dmult)
        if a not in id_to_idx: continue
        y = _parse_y(row["Y_vector"])
        aidx = id_to_idx[a];  xA = X32[aidx]

        idxM = _topM_idx(y, M)
        lb2 = _partial_lb_dist2(X32, xA, y, idxM)
        tested_total += X32.shape[0]
        keep = np.where(lb2 <= (D*D + EPS))[0]
        pruned_total += (X32.shape[0] - keep.size)
        if keep.size == 0:
            continue

        d2_sel = _w2_dist2(X32[keep], xA, y)
        mask = d2_sel <= (D*D + EPS)
        if not np.any(mask):
            continue
        keep = keep[mask]
        d2_sel = d2_sel[mask]

        order = np.argsort(d2_sel, kind="stable")
        keep = keep[order]; d2_sel = d2_sel[order]

        out.append(pd.DataFrame({
            "query_idx": qi, "point_A": a,
            "node": ids[keep], "dist": np.sqrt(d2_sel).astype(np.float32)
        }))

    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["query_idx","point_A","node","dist"])
    pruned_ratio = float(pruned_total / tested_total) if tested_total>0 else None
    return res, pruned_ratio

# ---------- ENDPOINTS ----------

# /search : exécution TP -> retourne TOUJOURS un CSV à télécharger (aucun JSON de résultats).
#   - mode : "naive" ou "pruned"
#   - topm : nb de features pour le test rapide (1..50) quand mode=pruned
#   - demo : 0 (TP normal: D du CSV) / 1 (Démo: D_eff = D*dscale, CSV inchangé)
#   - dscale : facteur pour la démo (ex. 0.6) ; ignoré si demo=0
@app.post("/search")
def search(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    mode: str = Query(default="pruned", pattern="^(naive|pruned)$"),
    topm: int = Query(default=12, ge=1, le=50),
    demo: int = Query(default=0, ge=0, le=1),
    dscale: float = Query(default=1.0, ge=0.01, le=1.0),
):
    try:
        gid, qid = BytesIO(graph_file.file.read()), BytesIO(queries_file.file.read())
        ids, X = _load_graph_from_file(gid)
        qid.seek(0);  qdf = _load_queries_from_file(qid)
        qdf = _map_pointA_to_graph_prefix(qdf, ids)

        dmult = float(dscale) if int(demo)==1 else 1.0

        if mode == "naive":
            res = _search_naive(ids, X, qdf, dmult)
            fname = "results_naive.csv"
        else:
            res, _ = _search_pruned_topM(ids, X, qdf, dmult, M=int(topm))
            fname = f"results_pruned_M{int(topm)}.csv"

        buf = StringIO();  res.to_csv(buf, index=False);  buf.seek(0)
        headers = {"Content-Disposition": f"attachment; filename={fname}",
                   "X-Row-Count": str(len(res))}
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# /stats : renvoie UNIQUEMENT des statistiques (pas de lignes) pour la comparaison visuelle.
@app.post("/stats")
def stats(
    graph_file: UploadFile = File(...),
    queries_file: UploadFile = File(...),
    mode: str = Query(default="pruned", pattern="^(naive|pruned)$"),
    topm: int = Query(default=12, ge=1, le=50),
    demo: int = Query(default=0, ge=0, le=1),
    dscale: float = Query(default=1.0, ge=0.01, le=1.0),
):
    try:
        gid, qid = BytesIO(graph_file.file.read()), BytesIO(queries_file.file.read())
        ids, X = _load_graph_from_file(gid)
        qid.seek(0);  qdf = _load_queries_from_file(qid)
        qdf = _map_pointA_to_graph_prefix(qdf, ids)

        dmult = float(dscale) if int(demo)==1 else 1.0

        t0 = time.perf_counter()
        if mode == "naive":
            res = _search_naive(ids, X, qdf, dmult)
            pruned_ratio = None
        else:
            res, pruned_ratio = _search_pruned_topM(ids, X, qdf, dmult, M=int(topm))
        t1 = time.perf_counter()

        return {
            "mode": mode,
            "topM": (None if mode=="naive" else int(topm)),
            "tp_mode": ("demo" if int(demo)==1 else "normal"),
            "n_nodes": len(ids),
            "n_queries": len(qdf),
            "n_rows": int(len(res)),
            "dscale_used": dmult,
            "algo_time_sec": round(t1 - t0, 6),
            "pruned_ratio": pruned_ratio
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ---------- UI : CSV only + bouton Comparer (visuel) ----------

@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!doctype html>
<html lang=fr><head><meta charset="utf-8"/><title>Recherche pondérée — CSV + Comparaison</title>
<style>
body{font-family:system-ui;max-width:900px;margin:40px auto;padding:0 16px}
.card{border:1px solid #ccc;border-radius:12px;padding:16px;margin:12px 0}
button{padding:10px 16px;border-radius:10px;border:1px solid #111;background:#111;color:#fff;cursor:pointer;margin-right:8px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
label{display:block;margin:8px 0 4px}
.bar{height:14px;background:#ddd;border-radius:7px;overflow:hidden}
.bar > div{height:100%}
</style>
</head><body>
<h1>Recherche pondérée — Export CSV & Comparaison (Top-M)</h1>

<form id=f onsubmit="run(event)">
  <div class=card>
    <label>Graph CSV</label>
    <input id=graph type=file accept=.csv required>

    <label>Queries CSV</label>
    <input id=queries type=file accept=.csv required>

    <div class=row style="margin-top:8px">
      <label>Mode</label>
      <select id=mode>
        <option value="naive">naive (exact)</option>
        <option value="pruned" selected>pruned (exact + Top-M)</option>
      </select>
      <label>Top-M</label>
      <input id=topm type=number min=1 max=50 step=1 value=12 style="width:90px">
    </div>

    <div class=card style="margin-top:12px;background:#fafafa">
      <strong>Démo (optionnel)</strong>
      <p>Pour visualiser l'accélération sans modifier vos CSV : <code>D_eff = D × dscale</code> pendant le calcul.</p>
      <div class=row>
        <label><input type=checkbox id=demo> Activer la démo</label>
        <label>Facteur dscale</label>
        <input id=dscale type=number min=0.10 max=1 step=0.05 value=0.60 style="width:120px">
      </div>
    </div>

    <div style="margin-top:10px">
      <button type="submit">Lancer et télécharger le CSV</button>
      <button type="button" onclick="compareRun()">Comparer naïve vs pruned (visuel)</button>
    </div>
  </div>
</form>

<section id=out class=card style="display:none"></section>

<script>
function bars(label, v1, v2){
  const mx=Math.max(v1,v2,0.001);
  const p1=Math.round(100*v1/mx), p2=Math.round(100*v2/mx);
  return `
    <div><strong>${label}</strong>
      <div class="bar"><div style="width:${p1}%;background:#4caf50"></div></div>
      <small>naive: ${v1.toFixed(4)}s</small>
      <div class="bar" style="margin-top:6px"><div style="width:${p2}%;background:#2196f3"></div></div>
      <small>pruned: ${v2.toFixed(4)}s</small>
    </div>`;
}

async function run(e){
  e.preventDefault();
  const graph=document.getElementById('graph').files[0];
  const queries=document.getElementById('queries').files[0];
  const mode=document.getElementById('mode').value;
  const topm=document.getElementById('topm').value || 12;
  const demo=document.getElementById('demo').checked ? 1 : 0;
  const dscale=document.getElementById('dscale').value || 1.0;
  if(!graph||!queries){ alert('Chargez les deux CSV.'); return; }

  const fd=new FormData();
  fd.append('graph_file',graph);
  fd.append('queries_file',queries);

  const url='/search?mode='+mode+'&topm='+topm+'&demo='+demo+'&dscale='+dscale;

  const res=await fetch(url,{method:'POST',body:fd});
  if(!res.ok){
    const j=await res.json().catch(()=>({error:'Erreur inconnue'}));
    alert('Erreur: '+(j.error||'')); return;
  }
  const blob=await res.blob();
  const disp = res.headers.get('Content-Disposition') || 'attachment; filename=results.csv';
  const fname = disp.split('filename=')[1] || 'results.csv';
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download=fname; document.body.appendChild(a); a.click(); a.remove();
}

async function fetchStats(mode, graph, queries, topm, demo, dscale){
  const fd=new FormData();
  fd.append('graph_file',graph);
  fd.append('queries_file',queries);
  const url='/stats?mode='+mode+'&topm='+topm+'&demo='+demo+'&dscale='+dscale;
  const res=await fetch(url,{method:'POST',body:fd});
  if(!res.ok){throw new Error((await res.json()).error||'Erreur stats');}
  return await res.json();
}

async function compareRun(){
  const out=document.getElementById('out');
  out.style.display='block';
  out.innerHTML='<em>Comparaison en cours…</em>';

  const graph=document.getElementById('graph').files[0];
  const queries=document.getElementById('queries').files[0];
  if(!graph||!queries){ out.innerHTML='<p>Charge d’abord les deux CSV.</p>'; return; }

  const topm=document.getElementById('topm').value || 12;
  const demo=document.getElementById('demo').checked ? 1 : 0;
  const dscale=document.getElementById('dscale').value || 1.0;

  try{
    const sN = await fetchStats('naive',  graph, queries, topm, demo, dscale);
    const sP = await fetchStats('pruned', graph, queries, topm, demo, dscale);

    const eq = (sN.n_rows===sP.n_rows) ? '✅ identiques' : `❌ différents (${sN.n_rows} vs ${sP.n_rows})`;
    const htmlBars = bars('Temps serveur (algo)', sN.algo_time_sec, sP.algo_time_sec);
    const pr = (sP.pruned_ratio!=null) ? (100*sP.pruned_ratio).toFixed(1)+'%' : '-';
    const speed = (sP.algo_time_sec>0) ? (sN.algo_time_sec/sP.algo_time_sec).toFixed(2) : '∞';

    out.innerHTML = `<h3>Comparaison naïve vs pruned (Top-M=${topm})</h3>
<pre>
Nœuds: ${sN.n_nodes} | Requêtes: ${sN.n_queries} | Mode TP: ${sN.tp_mode} | dscale: ${sN.dscale_used}
Lignes retournées: naive=${sN.n_rows}, pruned=${sP.n_rows} → ${eq}
</pre>
${htmlBars}
<pre>
Accélération (naïve/pruned): ×${speed}
% écartés (pruned, test rapide): ${pr}
</pre>`;
  }catch(e){
    out.innerHTML = `<pre>Erreur: ${e.message}</pre>`;
  }
}
</script>
</body></html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
