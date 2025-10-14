// TP Graph — Recherche & Bonus - Application JavaScript

// ========= Navigation =========

function show(id) {
    // Masquer toutes les sections
    document.getElementById('search').classList.add('hidden');
    document.getElementById('paths').classList.add('hidden');
    
    // Afficher la section demandée
    document.getElementById(id).classList.remove('hidden');
    
    // Mettre à jour les onglets actifs
    document.getElementById('tabS')?.classList.toggle('active', id === 'search');
    document.getElementById('tabP')?.classList.toggle('active', id === 'paths');
}

// ========= Gestion des algorithmes de chemins =========

function onAlgo() {
    const algo = document.getElementById('algo').value;
    const aParams = document.getElementById('aParams');
    const bParams = document.getElementById('bParams');
    
    aParams.style.display = (algo === 'astar') ? 'inline-flex' : 'none';
    bParams.style.display = (algo === 'beam') ? 'inline-flex' : 'none';
}

// Initialiser la gestion des algorithmes
document.addEventListener('DOMContentLoaded', function() {
    const algoSelect = document.getElementById('algo');
    if (algoSelect) {
        algoSelect.addEventListener('change', onAlgo);
        onAlgo(); // Initialiser l'état
    }
});

// ========= Recherche dans le rayon =========

async function runSearch(event) {
    event.preventDefault();
    
    // Récupérer les éléments du formulaire
    const graphFile = document.getElementById('g1').files[0];
    const queriesFile = document.getElementById('q1').files[0];
    const mode = document.getElementById('mode').value;
    const topM = parseInt(document.getElementById('topm').value || '12', 10);
    const dscale = parseFloat(document.getElementById('dscale').value || '1');
    const sinfo = document.getElementById('sinfo');
    
    // Validation
    if (!graphFile || !queriesFile) {
        sinfo.textContent = 'Veuillez sélectionner les deux fichiers CSV.';
        return;
    }
    
    sinfo.textContent = 'Calcul en cours…';
    
    try {
        // Préparer les données
        const formData = new FormData();
        formData.append('graph_file', graphFile);
        formData.append('queries_file', queriesFile);
        formData.append('mode', mode);
        formData.append('top_m', String(topM));
        formData.append('dscale', String(dscale));
        
        // Envoyer la requête
        const response = await fetch('/search_csv', {
            method: 'POST',
            body: formData
        });
        
        const blob = await response.blob();
        
        if (!response.ok) {
            const text = await blob.text();
            sinfo.textContent = `Erreur (${response.status}): ${text}`;
            return;
        }
        
        // Récupérer les métadonnées des headers
        const time = response.headers.get('X-Alg-Time');
        const rowCount = response.headers.get('X-Row-Count');
        const prunedRejects = response.headers.get('X-PrunedRejects');
        const matched = response.headers.get('X-Matched');
        const skipped = response.headers.get('X-Skipped');
        
        // Télécharger le fichier
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${mode}_results.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        // Afficher les statistiques
        let statsHTML = `
            <span class="badge">algo: ${Number(time).toFixed(3)}s</span>
            <span class="badge">lignes: ${rowCount}</span>
            <span class="badge">queries OK: ${matched}</span>
            <span class="badge">ignorées: ${skipped}</span>
        `;
        
        if (mode === 'pruned') {
            statsHTML += `<span class="badge">pruned rejects: ${prunedRejects}</span>`;
        }
        
        sinfo.innerHTML = statsHTML;
        
    } catch (error) {
        sinfo.textContent = `Erreur: ${error.message || error}`;
    }
}

async function runCompare() {
    const cinfo = document.getElementById('cinfo');
    const graphFile = document.getElementById('g1').files[0];
    const queriesFile = document.getElementById('q1').files[0];
    const topM = parseInt(document.getElementById('topm').value || '12', 10);
    const dscale = parseFloat(document.getElementById('dscale').value || '1');
    
    // Validation
    if (!graphFile || !queriesFile) {
        cinfo.classList.remove('hidden');
        cinfo.textContent = 'Veuillez d\'abord charger les deux fichiers CSV.';
        return;
    }
    
    cinfo.classList.remove('hidden');
    cinfo.textContent = 'Comparaison en cours…';
    
    try {
        // Préparer les données
        const formData = new FormData();
        formData.append('graph_file', graphFile);
        formData.append('queries_file', queriesFile);
        formData.append('top_m', String(topM));
        formData.append('dscale', String(dscale));
        
        // Envoyer la requête
        const response = await fetch('/compare_search', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok || result.error) {
            cinfo.textContent = `Erreur (${response.status}): ${result.error}`;
            return;
        }
        
        // Afficher les résultats de comparaison
        const equality = result.equal ? '✅ identiques' : '❌ différents';
        
        cinfo.textContent = `naïf: ${result.rows_naive} lignes, ${result.naive_time_sec.toFixed(3)}s
pruned: ${result.rows_pruned} lignes, ${result.pruned_time_sec.toFixed(3)}s
pruned rejects: ${result.pruned_rejects}
queries OK: ${result.matched_queries} | ignorées: ${result.skipped_queries}
égalité: ${equality}`;
        
    } catch (error) {
        cinfo.textContent = `Erreur: ${error.message || error}`;
    }
}

// ========= Calcul de chemins =========

async function runPath(event) {
    event.preventDefault();
    
    // Récupérer les éléments du formulaire
    const pout = document.getElementById('pout');
    const graphFile = document.getElementById('g2').files[0];
    const queriesFile = document.getElementById('q2').files[0];
    const src = document.getElementById('src').value.trim();
    const dst = document.getElementById('dst').value.trim();
    const yVector = document.getElementById('yvec').value.trim();
    const algo = document.getElementById('algo').value;
    
    // Validation
    if (!graphFile) {
        pout.textContent = 'Veuillez sélectionner un fichier graphe CSV.';
        return;
    }
    
    if (!src || !dst) {
        pout.textContent = 'Veuillez spécifier les nœuds A et B.';
        return;
    }
    
    pout.textContent = 'Calcul en cours…';
    
    try {
        // Préparer les données
        const formData = new FormData();
        formData.append('graph_file', graphFile);
        if (queriesFile) {
            formData.append('queries_file', queriesFile);
        }
        
        // Construire l'URL selon l'algorithme
        let url = '';
        if (algo === 'astar') {
            const k = parseInt(document.getElementById('kA').value || '10', 10);
            const maxExp = parseInt(document.getElementById('maxexp').value || '20000', 10);
            url = `/path_astar?src=${encodeURIComponent(src)}&dst=${encodeURIComponent(dst)}&k=${k}&maxexp=${maxExp}`;
        } else {
            const kNeighbors = parseInt(document.getElementById('kB').value || '10', 10);
            const K = parseInt(document.getElementById('K').value || '6', 10);
            const beamWidth = parseInt(document.getElementById('BW').value || '16', 10);
            url = `/path_beam?src=${encodeURIComponent(src)}&dst=${encodeURIComponent(dst)}&k_neighbors=${kNeighbors}&K=${K}&beam=${beamWidth}`;
        }
        
        // Ajouter le vecteur Y si spécifié
        if (yVector) {
            url += `&y_vector=${encodeURIComponent(yVector)}`;
        }
        
        // Envoyer la requête
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        const rawText = await response.text();
        let result;
        
        try {
            result = JSON.parse(rawText);
        } catch {
            result = { error: 'Réponse non-JSON', body: rawText };
        }
        
        if (!response.ok || result.error) {
            pout.textContent = `Erreur (${response.status}): ${result.error || result.body}`;
            return;
        }
        
        // Afficher les résultats
        const path = (result.path || []).join(' → ');
        const algoType = result.exact ? 'A* (exact)' : 'Beam (heuristique)';
        const note = result.note ? (' | ' + result.note) : '';
        
        pout.textContent = `Algo: ${algoType} | nœuds: ${result.n_nodes} | expansions: ${result.expanded}
Coût total: ${result.cost != null ? result.cost.toFixed(6) : '-'} | Temps algo: ${result.algo_time_sec ? result.algo_time_sec.toFixed(3) + 's' : '-'}
Chemin: ${path || '-'}${note}`;
        
    } catch (error) {
        pout.textContent = `Erreur: ${error.message || error}`;
    }
}
