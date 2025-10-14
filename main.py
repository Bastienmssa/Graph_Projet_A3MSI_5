#!/usr/bin/env python3
"""
TP Graph — Recherche & Bonus
Point d'entrée principal pour lancer l'application web.

Usage:
    python main.py [--host HOST] [--port PORT] [--reload]

Exemples:
    python main.py                          # Lancement standard (localhost:8000)
    python main.py --port 3000             # Port personnalisé
    python main.py --host 0.0.0.0 --reload # Mode développement avec rechargement auto
"""

import os
import sys
import argparse
import uvicorn
import numpy as np

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="TP Graph — Recherche & Bonus - Serveur Web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Adresse IP d'écoute (défaut: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("PORT", "8000")),
        help="Port d'écoute (défaut: 8000, ou variable PORT)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Mode développement avec rechargement automatique"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Niveau de log (défaut: info)"
    )
    
    return parser.parse_args()

def setup_environment():
    """Configure l'environnement d'exécution."""
    # Fixer la graine aléatoire pour la reproductibilité
    np.random.seed(42)
    
    # Vérifier que les modules nécessaires sont disponibles
    try:
        from api import app
        return app
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Vérifiez que tous les modules sont présents:")
        print("  - api.py")
        print("  - graph_build.py") 
        print("  - weighted_distance.py")
        print("  - radiusx_search.py")
        sys.exit(1)

def main():
    """Point d'entrée principal."""
    args = parse_arguments()
    
    print("🚀 TP Graph — Recherche & Bonus")
    print("=" * 50)
    
    # Configuration de l'environnement
    app = setup_environment()
    
    # Informations de démarrage
    print(f"📡 Serveur: http://{args.host}:{args.port}")
    print(f"🔧 Mode: {'Développement' if args.reload else 'Production'}")
    print(f"📊 Log level: {args.log_level}")
    
    if args.reload:
        print("⚡ Rechargement automatique activé")
    
    print("=" * 50)
    print("Appuyez sur Ctrl+C pour arrêter le serveur")
    print()
    
    try:
        # Lancement du serveur
        uvicorn.run(
            "api:app",  # Module:variable
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Arrêt du serveur...")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
