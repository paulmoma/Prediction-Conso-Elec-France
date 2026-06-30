#!/bin/bash
# cron_setup.sh : configure les cron jobs automatiquement

PROJECT_DIR=$(pwd)
PYTHON="$HOME/miniforge3/envs/prophet/bin/python"

echo "Projet    : $PROJECT_DIR"
echo "Python    : $PYTHON"

if [ ! -x "$PYTHON" ]; then
    echo "❌ Python introuvable : $PYTHON"
    echo "   Vérifier le nom de l'env mamba (actuellement 'prophet')"
    exit 1
fi

chmod +x "$PROJECT_DIR/run_if_needed.sh"

STREAMLIT_URL="https://prediction-conso-elec-france-szs2wnajdluqim2ybsxxk4.streamlit.app/"

# Prévisions + validation + réentraînement : lundi et jeudi à 10h
(crontab -l 2>/dev/null; echo "# Energy Forecasting : run hebdomadaire") | crontab -
(crontab -l 2>/dev/null; echo "0 10 * * 1,4 $PROJECT_DIR/run_if_needed.sh") | crontab -

# Ping 10h et 16h pour garder l'app Streamlit Cloud active (évite le cold start)
(crontab -l 2>/dev/null; echo "# Energy Forecasting : ping Streamlit quotidien") | crontab -
(crontab -l 2>/dev/null; echo "0 10,16 * * * curl -s '$STREAMLIT_URL' > /dev/null 2>&1") | crontab -

echo ""
echo "✅ Cron configuré :"
crontab -l | grep -A1 "Energy Forecasting"
