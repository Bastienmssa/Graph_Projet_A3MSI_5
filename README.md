# Graph_Projet_A3MSI_5

## 📊 Simulation de recherche publicitaire sur graphe pondéré

**A3MSI Groupe 5** - Projet d'algorithmique sur graphes

---

## 🎯 Description du projet

Ce projet implémente un système de **recherche publicitaire** sur un graphe pondéré, permettant de :

- **Rechercher des nœuds similaires** dans un graphe en fonction de critères pondérés
- **Optimiser les performances** avec des algorithmes naïfs et pruned (Top-M)
- **Calculer des chemins optimaux** entre nœuds avec A* et Beam Search
- **Comparer les performances** entre différentes approches algorithmiques

## 🚀 Fonctionnalités principales

### 🔍 Recherche de similarité
- **Algorithme naïf** : calcul complet de distance sur toutes les features
- **Algorithme pruned** : optimisation avec borne inférieure sur les M features les plus importantes
- **Distance pondérée** : `d²(A,B) = Σ(y_i × (x_A,i - x_B,i)²)`

### 🛤️ Calcul de chemins
- **A* exact** : recherche de chemin optimal avec heuristique admissible
- **Beam Search** : approche heuristique plus rapide pour de gros graphes
- **Voisinage dynamique** : k-NN calculé à la volée selon les poids Y

### 📈 Interface web interactive
- Interface FastAPI avec documentation automatique
- Upload de fichiers CSV pour graphe et requêtes
- Comparaison en temps réel des performances
- Export des résultats en CSV

## 🏗️ Architecture technique

### Structure des données
```
Data/
├── adsSim_data_nodes.csv    # Graphe : nœuds avec 50 features + cluster_id
├── queries_structured.csv   # Requêtes : point_A, Y_vector, D
├── small_data.csv          # Jeu de données réduit pour tests
└── test_*.xlsx            # Fichiers de test Excel
```

### Format des données

**Graphe (adsSim_data_nodes.csv)** :
- `node_id` : identifiant du nœud (ex: node_1, node_2...)
- `feature_1` à `feature_50` : 50 caractéristiques numériques par nœud
- `cluster_id` : identifiant du cluster (optionnel)

**Requêtes (queries_structured.csv)** :
- `point_A` : point de départ (nœud existant ou nouveau point)
- `Y_vector` : 50 poids séparés par ';' pour la distance pondérée
- `D` : distance maximale de recherche

## 🛠️ Installation et utilisation

### Prérequis
```bash
Python 3.8+
```

### Installation
```bash
# Cloner le projet
git clone <repository-url>
cd Graph_Projet_A3MSI_5

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement du serveur
```bash
python final.py
```

Le serveur sera accessible sur : `http://localhost:8000`

### API Endpoints

#### 🔍 Recherche
- `POST /search_csv` : Recherche détaillée avec résultats complets
- `POST /search_summary_csv` : Résumé avec 1 ligne par requête
- `POST /compare_search` : Comparaison naïf vs pruned

#### 🛤️ Chemins
- `POST /path_astar` : Chemin optimal avec A*
- `POST /path_beam` : Chemin heuristique avec Beam Search

## 📊 Exemples d'utilisation

### Recherche de similarité
```python
# Paramètres
mode = "pruned"        # ou "naive"
top_m = 12            # nombre de features pour la borne
dscale = 1.0          # facteur d'échelle pour D

# Résultat : nœuds avec distance ≤ D
```

### Calcul de chemin
```python
# A* exact
src = "node_1"
dst = "node_100"
k = 10                # voisins par nœud
max_expansions = 20000

# Beam Search
K = 6                 # profondeur maximale
beam_width = 16       # largeur du faisceau
```

## 🧮 Algorithmes implémentés

### Recherche pruned (Top-M)
1. **Sélection** : identifier les M features avec les plus gros poids Y
2. **Borne inférieure** : calculer `lb² = Σ(y_top × (x_A,top - x_B,top)²)`
3. **Filtrage** : garder seulement les nœuds avec `lb² ≤ D²`
4. **Distance exacte** : calculer la distance complète sur les candidats

### A* avec voisinage dynamique
1. **Heuristique** : distance pondérée directe vers la cible
2. **Voisinage** : k-NN recalculé selon les poids Y actuels
3. **Exploration** : priorité aux nœuds avec f = g + h minimal

### Beam Search K-sauts
1. **Exploration limitée** : K niveaux maximum
2. **Faisceau** : garder seulement les meilleurs chemins
3. **Heuristique** : pas de garantie d'optimalité

## 📈 Performances

### Optimisations implémentées
- **Vectorisation NumPy** : calculs matriciels optimisés
- **Filtrage précoce** : élimination rapide des candidats
- **Cache des distances** : réutilisation des calculs
- **Types optimisés** : float32 pour réduire la mémoire

### Comparaison des algorithmes
- **Naïf** : O(N) par requête, simple mais coûteux
- **Pruned** : O(N) dans le pire cas, mais très rapide en pratique
- **A*** : O(b^d) avec b = branching factor, d = profondeur
- **Beam** : O(K × beam_width × k) par niveau

## 📁 Structure du projet

```
Graph_Projet_A3MSI_5/
├── final.py                 # Application FastAPI principale
├── requirements.txt         # Dépendances Python
├── README.md               # Documentation
├── Data/                   # Jeux de données
│   ├── adsSim_data_nodes.csv
│   ├── queries_structured.csv
│   ├── small_data.csv
│   └── test_*.xlsx
└── __pycache__/           # Cache Python
```

## 🔧 Dépendances

```
fastapi>=0.104.0          # Framework web
uvicorn>=0.24.0           # Serveur ASGI
numpy>=1.24.0             # Calculs numériques
pandas>=2.0.0             # Manipulation de données
python-multipart>=0.0.6   # Upload de fichiers
```

## 🎓 Contexte académique

**Matière** : Algorithmique sur graphes  
**Niveau** : A3MSI (3ème année)  
**Groupe** : 5  
**Objectif** : Implémentation d'algorithmes de recherche et de chemin sur graphes pondérés

## 📝 Notes techniques

- **Distance pondérée** : utilise la norme L2 pondérée par le vecteur Y
- **Compatibilité** : support des formats CSV avec détection automatique des colonnes
- **Robustesse** : gestion d'erreurs et validation des données d'entrée
- **Interface** : documentation automatique avec Swagger UI

---

*Développé dans le cadre du cours d'algorithmique sur graphes - ESME*
