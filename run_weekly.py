"""
run_weekly.py
Pipeline bi-hebdomadaire : lundi et jeudi à 10h.

Étape 1 : Valide les prévisions passées vs données RTE réelles
Étape 2 : Charge le modèle Production du registry + génère les nouvelles prévisions
Étape 3 : Appelle retrain.py pour réentraîner et évaluer la promotion en Production

Fallback : si aucun modèle Production n'existe encore dans le registry, entraîne à la volée.

Cron : 0 10 * * 1,4 cd /path/to/project && /path/to/python run_weekly.py >> logs/pipeline.log 2>&1
MLflow UI : mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
"""

import logging
import mlflow
import mlflow.prophet
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from src.data import (get_temperature_weighted, get_temperature_forecast,
                       build_df_model, validate_temperature, POINTS_RURAUX,
                       load_rte_complete)
from src.features import make_all_features
from src.model import predict, train
import retrain as retrain_module

# Configuration
DATA_DIR        = Path('data')
FORECAST_DIR    = DATA_DIR / 'forecasts'
LOG_DIR         = Path('logs')
MLFLOW_DB       = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'EF_Weekly'
TRAIN_START     = '2023-01-01'
MAPE_ALERT_PCT  = 5.0
MODEL_NAMES     = {'7j': 'prophet_7j', '30j': 'prophet_30j'}

FORECAST_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s %(levelname)s %(message)s',
    handlers= [
        logging.FileHandler(LOG_DIR / 'pipeline.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# Étape 1 : Validation rétrospective

def find_last_forecast(model: str) -> Path | None:
    """Retourne le fichier de prévision le plus récent (hors aujourd'hui)."""
    files = sorted(FORECAST_DIR.glob(f'forecast_{model}_*.csv'))
    past  = [f for f in files if f.stem.split('_')[-1] < str(date.today())]
    return past[-1] if past else None


def validate_past_forecasts(df_rte: pd.DataFrame) -> dict:
    """Compare les prévisions du run précédent aux données RTE réelles, met à jour validation_log.csv."""
    results = {}

    for model in ['7j', '30j']:
        last_file = find_last_forecast(model)
        if not last_file:
            logger.info(f"[{model}] Aucune prévision passée trouvée")
            continue

        run_date = last_file.stem.split('_')[-1]
        logger.info(f"[{model}] Validation du run {run_date}")

        df_fc  = pd.read_csv(last_file, parse_dates=['ds'])
        df_val = df_fc.merge(df_rte[['ds', 'y']], on='ds', how='inner').dropna()

        if len(df_val) == 0:
            logger.warning(f"[{model}] Pas encore de données RTE pour valider")
            continue

        mape   = mean_absolute_percentage_error(df_val['y'].values, df_val['yhat'].values) * 100
        mae    = mean_absolute_error(df_val['y'].values, df_val['yhat'].values)
        status = '⚠️  ALERTE' if mape > MAPE_ALERT_PCT else '✅ OK'

        logger.info(f"[{model}] MAPE={mape:.2f}%  MAE={mae:,.0f}MW  "
                    f"({len(df_val)} jours validés)  {status}")

        if mape > MAPE_ALERT_PCT:
            logger.warning(f"[{model}] MAPE > {MAPE_ALERT_PCT}% : "
                           f"vérifier anomalie météo ou comportement atypique")

        results[model] = {
            'run_date': run_date,
            'n_days'  : len(df_val),
            'mape'    : round(mape, 3),
            'mae'     : round(mae, 1),
            'status'  : status,
        }

        # Journal historique CSV
        log_path = DATA_DIR / 'validation_log.csv'
        row = pd.DataFrame([{
            'validation_date': str(date.today()),
            'run_date'       : run_date,
            'model'          : model,
            'mape'           : round(mape, 3),
            'mae'            : round(mae, 1),
            'n_days'         : len(df_val),
            'alert'          : mape > MAPE_ALERT_PCT,
        }])
        if log_path.exists():
            pd.concat([pd.read_csv(log_path), row]).to_csv(log_path, index=False)
        else:
            row.to_csv(log_path, index=False)

    return results


# Étape 2 : Prévisions avec le modèle Production

def _log_temperature_forecast(df_temp_fc: pd.DataFrame, run_date: str) -> None:
    """Ajoute les prévisions de températures au journal historique temp_forecast_log.csv."""
    log_path = DATA_DIR / 'temp_forecast_log.csv'

    df_log = df_temp_fc[['ds', 'temp', 'temp_min', 'temp_max', 'source']].copy()
    df_log.insert(0, 'run_date', run_date)
    df_log['ds'] = df_log['ds'].dt.strftime('%Y-%m-%d')

    if log_path.exists():
        df_existing = pd.read_csv(log_path)
        df_existing = df_existing[df_existing['run_date'] != run_date]
        df_log = pd.concat([df_existing, df_log]).reset_index(drop=True)

    df_log.to_csv(log_path, index=False)
    logger.info(f"Prévisions températures journalisées → {log_path.name} "
                f"({len(df_temp_fc)} jours, run_date={run_date})")


def build_feature_dfs(df_rte: pd.DataFrame,
                       today: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit les DataFrames features 7j et 30j, températures historiques + prévisions."""
    df_temp_hist = get_temperature_weighted(
        start=TRAIN_START, end=today,
        points=POINTS_RURAUX, cache_dir=DATA_DIR
    )
    validate_temperature(df_temp_hist, 'hist')
    df_temp_hist.to_csv(DATA_DIR / 'temp_history.csv', index=False)

    df_temp_fc = get_temperature_forecast(horizon_days=30, cache_dir=DATA_DIR)
    df_temp_fc.to_csv(DATA_DIR / 'temperature_forecast.csv', index=False)
    _log_temperature_forecast(df_temp_fc, today)

    # Enlève la colonne source avant le feature engineering (non numérique)
    df_temp_fc_model = df_temp_fc.drop(columns=['source'])

    df_model = build_df_model(df_rte, df_temp_hist)

    # DataFrame des jours futurs (sans y)
    df_future_temp = build_df_model(
        pd.DataFrame({'ds': df_temp_fc_model['ds'], 'y': np.nan}),
        df_temp_fc_model
    )

    df_7j  = make_all_features(df_model, model='7j')
    df_30j = make_all_features(df_model, model='30j')

    df_7j_full  = pd.concat([df_7j,  make_all_features(df_future_temp, model='7j')]
                             ).reset_index(drop=True)
    df_30j_full = pd.concat([df_30j, make_all_features(df_future_temp, model='30j')]
                             ).reset_index(drop=True)

    return df_7j_full, df_30j_full


def load_production_models() -> tuple:
    """Charge les modèles Production depuis le registry. Retourne (m_7j, m_30j, source)."""
    try:
        m_7j  = mlflow.prophet.load_model(f"models:/{MODEL_NAMES['7j']}/Production")
        m_30j = mlflow.prophet.load_model(f"models:/{MODEL_NAMES['30j']}/Production")
        logger.info("Modèles chargés depuis le registry (Production)")
        return m_7j, m_30j, 'registry'
    except Exception as e:
        logger.warning(f"Registry indisponible ({e})")
        logger.warning("Fallback : entraînement à la volée : "
                       "lancer retrain.py pour initialiser le registry")
        return None, None, 'retrain'


def save_dated_forecasts(fc_7j: pd.DataFrame, fc_30j: pd.DataFrame) -> None:
    """Sauvegarde les prévisions avec horodatage + version latest pour le dashboard."""
    today = str(date.today())
    today_ts = pd.Timestamp(today)
    for model, fc, h in [('7j', fc_7j, 7), ('30j', fc_30j, 30)]:
        cols        = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        out         = fc[fc['ds'] >= today_ts].head(h)[cols]
        dated_path  = FORECAST_DIR / f'forecast_{model}_{today}.csv'
        latest_path = DATA_DIR     / f'forecast_{model}_latest.csv'
        out.to_csv(dated_path,  index=False)
        out.to_csv(latest_path, index=False)
        logger.info(f"[{model}] Prévision sauvegardée → {dated_path.name}")


# Run principal

def main():
    today = str(date.today())
    logger.info(f"{'='*55}")
    logger.info(f"  Run hebdomadaire : {today}")
    logger.info(f"{'='*55}")

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"weekly_{today}") as run_ctx:

        # Données RTE : XLS historiques + extension API (via load_rte_complete)
        logger.info("Chargement données RTE...")
        df_rte = load_rte_complete(DATA_DIR)

        # Étape 1 : validation rétrospective
        logger.info("── ÉTAPE 1 : Validation rétrospective ──")
        validation = validate_past_forecasts(df_rte)

        retro_metrics = {}
        all_ok        = True
        for model, m in validation.items():
            retro_metrics[f'retro_mape_{model}'] = m['mape']
            retro_metrics[f'retro_mae_{model}']  = m['mae']
            retro_metrics[f'retro_ndays_{model}'] = m['n_days']
            if m['mape'] > MAPE_ALERT_PCT:
                all_ok = False

        if retro_metrics:
            mlflow.log_metrics(retro_metrics)

        # Étape 2 : nouvelles prévisions
        logger.info("── ÉTAPE 2 : Nouvelles prévisions ──")

        logger.info("Construction des features...")
        df_7j_full, df_30j_full = build_feature_dfs(df_rte, today)

        m_7j, m_30j, model_source = load_production_models()

        if model_source == 'retrain':
            # Fallback : entraîner si le registry est vide (premier run)
            m_7j  = train(df_7j_full,  model='7j',  train_start=TRAIN_START)
            m_30j = train(df_30j_full, model='30j', train_start=TRAIN_START)

        fc_7j  = predict(m_7j,  df_7j_full,  model='7j',  horizon=7)
        fc_30j = predict(m_30j, df_30j_full, model='30j', horizon=30)

        save_dated_forecasts(fc_7j, fc_30j)

        # Tags de synthèse
        mlflow.set_tags({
            'run_date'    : today,
            'model_source': model_source,
            'retro_status': '✅ OK' if all_ok else '⚠️  ALERTE',
            'n_validated' : str(len(validation)),
            'status'      : '✅' if all_ok else '⚠️',
        })

        logger.info(f"{'='*55}")
        logger.info(f"  Run terminé : {run_ctx.info.run_id[:8]} : "
                    f"{'✅ OK' if all_ok else '⚠️  ALERTE'}")
        logger.info(f"{'='*55}")

    # Étape 3 : réentraînement
    logger.info("── ÉTAPE 3 : Réentraînement ──")
    retrain_module.run()


if __name__ == '__main__':
    main()
