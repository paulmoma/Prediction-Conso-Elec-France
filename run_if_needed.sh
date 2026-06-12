#!/bin/bash
# run_if_needed.sh : Lance run_weekly.py seulement si pas encore tourné aujourd'hui.
# Utilisé par le cron lundi+jeudi 10h. Un run manuel préalable (qui crée le CSV)
# suffit à court-circuiter le cron.

cd "$(dirname "$0")"   # se place dans le répertoire du projet quelle que soit l'origine

TODAY=$(date +%Y-%m-%d)
FORECAST="data/forecasts/forecast_7j_${TODAY}.csv"
LOG="logs/pipeline.log"

mkdir -p logs

if [ -f "$FORECAST" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') INFO  Cron skip : prévision ${TODAY} déjà présente" >> "$LOG"
    exit 0
fi

# Chemin direct vers Python de l'env mamba/conda (pas besoin d'activation)
PYTHON="$HOME/miniforge3/envs/prophet/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR Python introuvable : $PYTHON" >> "$LOG"
    exit 1
fi

"$PYTHON" run_weekly.py
