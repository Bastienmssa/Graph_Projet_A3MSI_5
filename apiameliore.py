from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
import re, time

# Import des modules dédiés
from load_data import (_detect_id_col, _load_graph_from_file, _load_queries_from_file, 
                       _parse_y, _graph_prefix, _map_pointA_to_graph_prefix)
from weighted_distance import _w2_dist2
from radiusx_search import _search_naive, _search_pruned_topM, _topM_idx, _partial_lb_dist2

app = FastAPI(title="Recherche pondérée — CSV + Comparaison (naive vs pruned Top-M)", version="3.0")

# ---------- Fonctions de recherche dans un rayon ----------
# Toutes les fonctions de l'Étape 3 déplacées vers radiusx_search.py

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
