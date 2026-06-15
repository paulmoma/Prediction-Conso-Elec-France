"""
retrain.py
Cycle de réentraînement avec holdout 8 semaines.

1. Entraîne sur [TRAIN_START → aujourd'hui - 8 semaines]
2. Évalue sur les 8 dernières semaines (MAPE + test overfitting Mann-Whitney)
3. Si les métriques passent → enregistre dans MLflow Model Registry + promeut en Production
4. Si échec → conserve le modèle Production actuel, run tagué 'rejected'

Usage:
    python retrain.py            # évalue + promeut si OK
    python retrain.py --dry-run  # évalue sans promouvoir en Production
"""

import argparse
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path

import mlflow
import mlflow.prophet
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from src.data import (get_temperature_weighted, get_temperature_forecast,
                       build_df_model, validate_temperature, POINTS_RURAUX,
                       load_rte_complete)
from src.features import (make_all_features, BEST_PARAMS_7J, BEST_PARAMS_30J)
from src.model import train, predict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR        = Path('data')
MLFLOW_DB       = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'EF_Retraining'
TRAIN_START     = '2023-01-01'
HOLD_OUT_WEEKS  = 8

# Seuil de MAPE au-delà duquel on ne promeut pas
MAPE_MAX_PROMOTE = 5.0
# Test d'overfitting
OVERFIT_ALPHA   = 0.05
OVERFIT_GAP_MAX = 0.40

# Noms dans le Model Registry
MODEL_NAMES = {'7j': 'prophet_7j', '30j': 'prophet_30j'}

# Baseline MAPE de référence pour le test d'overfitting (Mann-Whitney).
# Valeurs Optuna CV biaisées par sélection → remplacées par les holdouts observés
# en production (estimation non biaisée de la performance réelle).
MAPE_CV_FOLDS = {
    '7j' : [3.09] * 12,
    '30j': [2.78] * 12,
}


# Helpers évaluation

def rolling_weekly_mape(df_real: pd.DataFrame,
                         fc: pd.DataFrame,
                         start: str,
                         end: str) -> list[dict]:
    """MAPE et MAE semaine par semaine sur la période [start, end]."""
    weeks   = pd.date_range(start, end, freq='W-MON')
    results = []
    for w in weeks:
        w_end  = w + timedelta(days=6)
        df_w   = (df_real[df_real['ds'].between(str(w), str(w_end))][['ds', 'y']]
                  .merge(fc[['ds', 'yhat']], on='ds', how='inner')
                  .dropna())
        real   = df_w['y'].values
        pred   = df_w['yhat'].values
        if len(real) < 5:
            continue
        mape = mean_absolute_percentage_error(real, pred) * 100
        mae  = mean_absolute_error(real, pred)
        results.append({'week_start': w, 'mape': mape, 'mae': mae})
    return results


def test_overfitting(mape_cv_folds: list,
                     mape_holdout_weeks: list,
                     alpha: float = OVERFIT_ALPHA,
                     gap_threshold: float = OVERFIT_GAP_MAX) -> dict:
    """Mann-Whitney U + écart relatif. Overfitting = test significatif ET écart > gap_threshold."""
    mape_cv      = np.mean(mape_cv_folds)
    mape_holdout = np.mean(mape_holdout_weeks)
    gap          = (mape_holdout - mape_cv) / mape_cv

    _, p_value      = stats.mannwhitneyu(mape_holdout_weeks, mape_cv_folds,
                                          alternative='greater')
    overfit_alert   = (p_value < alpha) and (gap > gap_threshold)

    result = {
        'mape_cv'      : round(mape_cv, 3),
        'mape_holdout' : round(mape_holdout, 3),
        'gap_pct'      : round(gap * 100, 1),
        'p_value'      : round(p_value, 4),
        'overfit_alert': overfit_alert,
    }
    logger.info(
        f"Overfitting : CV={mape_cv:.2f}% holdout={mape_holdout:.2f}% "
        f"gap={gap*100:+.1f}% p={p_value:.4f} → "
        f"{'⚠️  OVERFIT' if overfit_alert else '✅ OK'}"
    )
    return result


def should_promote(mape_holdout: float, overfit: dict) -> tuple[bool, str]:
    """Retourne (True, raison) si le modèle peut être promu en Production."""
    if overfit['overfit_alert']:
        return False, f"overfitting détecté (gap={overfit['gap_pct']:+.1f}% p={overfit['p_value']:.4f})"
    if mape_holdout > MAPE_MAX_PROMOTE:
        return False, f"MAPE holdout {mape_holdout:.2f}% > seuil {MAPE_MAX_PROMOTE}%"
    return True, f"MAPE={mape_holdout:.2f}% OK, pas d'overfitting"


# Promotion dans le Model Registry

def promote_to_production(client: mlflow.tracking.MlflowClient,
                           model_name: str,
                           mape: float,
                           run_id: str) -> str:
    """Archive les versions Production existantes et promeut la dernière version en attente."""
    for v in client.get_latest_versions(model_name, stages=['Production']):
        client.transition_model_version_stage(model_name, v.version, 'Archived')
        client.set_model_version_tag(model_name, v.version, 'stage', 'Archived')
        client.set_model_version_tag(model_name, v.version, 'archived_date', str(date.today()))
        logger.info(f"[{model_name}] v{v.version} → Archived")

    candidates = client.get_latest_versions(model_name, stages=['None'])
    if not candidates:
        raise RuntimeError(f"Aucune version en attente pour {model_name}")

    new_version = max(candidates, key=lambda v: int(v.version)).version

    client.transition_model_version_stage(model_name, new_version, 'Production')
    client.update_model_version(
        model_name, new_version,
        description=f"MAPE holdout 8s={mape:.2f}% | run={run_id[:8]} | {date.today()}"
    )
    client.set_model_version_tag(model_name, new_version, 'stage', 'Production ✅')
    client.set_model_version_tag(model_name, new_version, 'promoted_date', str(date.today()))
    client.set_model_version_tag(model_name, new_version, 'mape_holdout', f'{mape:.2f}%')
    logger.info(f"[{model_name}] v{new_version} → Production ✅")
    return new_version


# Pipeline de réentraînement

def run(dry_run: bool = False):
    today          = str(date.today())
    hold_out_start = str(date.today() - timedelta(weeks=HOLD_OUT_WEEKS))
    logger.info(f"{'='*60}")
    logger.info(f"  Retrain {today} | train→{hold_out_start} | holdout {hold_out_start}→{today}")
    if dry_run:
        logger.info("  MODE DRY-RUN : aucune promotion ne sera effectuée")
    logger.info(f"{'='*60}")

    # 1. Données températures
    logger.info("Chargement températures historiques...")
    df_temp_hist = get_temperature_weighted(
        start=TRAIN_START, end=today,
        points=POINTS_RURAUX, cache_dir=DATA_DIR
    )
    validate_temperature(df_temp_hist, 'hist')

    # 2. Données consommation RTE (XLS + extension API, même source que run_weekly)
    logger.info("Chargement consommation RTE...")
    df_daily = load_rte_complete(DATA_DIR)

    # 3. Fusion + feature engineering
    df_model = build_df_model(df_daily, df_temp_hist)
    df_7j    = make_all_features(df_model, model='7j')
    df_30j   = make_all_features(df_model, model='30j')

    # 4. Entraînement sur [TRAIN_START → hold_out_start]
    logger.info(f"Entraînement sur [TRAIN_START → {hold_out_start}]...")
    m_7j  = train(df_7j,  model='7j',  train_start=TRAIN_START, train_end=hold_out_start)
    m_30j = train(df_30j, model='30j', train_start=TRAIN_START, train_end=hold_out_start)

    # 5. Prévisions sur le holdout (8 semaines = 56 jours)
    logger.info("Prévisions sur la période holdout...")
    fc_7j  = predict(m_7j,  df_7j,  model='7j',  horizon=HOLD_OUT_WEEKS * 7)
    fc_30j = predict(m_30j, df_30j, model='30j', horizon=HOLD_OUT_WEEKS * 7)

    # 6. Évaluation semaine par semaine
    logger.info("Évaluation holdout...")
    weeks_7j  = rolling_weekly_mape(df_7j,  fc_7j,  hold_out_start, today)
    weeks_30j = rolling_weekly_mape(df_30j, fc_30j, hold_out_start, today)

    if not weeks_7j or not weeks_30j:
        raise RuntimeError(
            "Pas assez de semaines complètes dans le holdout pour évaluer. "
            "Vérifier les données RTE."
        )

    mape_7j  = np.mean([w['mape'] for w in weeks_7j])
    mape_30j = np.mean([w['mape'] for w in weeks_30j])
    mae_7j   = np.mean([w['mae']  for w in weeks_7j])
    mae_30j  = np.mean([w['mae']  for w in weeks_30j])

    logger.info(f"MAPE holdout 7j  : {mape_7j:.2f}%  MAE={mae_7j:,.0f}MW  ({len(weeks_7j)} semaines)")
    logger.info(f"MAPE holdout 30j : {mape_30j:.2f}%  MAE={mae_30j:,.0f}MW  ({len(weeks_30j)} semaines)")

    # 7. Test overfitting
    overfit_7j  = test_overfitting(MAPE_CV_FOLDS['7j'],
                                    [w['mape'] for w in weeks_7j])
    overfit_30j = test_overfitting(MAPE_CV_FOLDS['30j'],
                                    [w['mape'] for w in weeks_30j])

    # 8. Décision de promotion (les deux modèles ensemble ou aucun)
    ok_7j,  reason_7j  = should_promote(mape_7j,  overfit_7j)
    ok_30j, reason_30j = should_promote(mape_30j, overfit_30j)
    promote = ok_7j and ok_30j

    if promote:
        logger.info("✅ Les deux modèles passent les critères → promotion en Production")
    else:
        reasons = []
        if not ok_7j:
            reasons.append(f"7j : {reason_7j}")
        if not ok_30j:
            reasons.append(f"30j : {reason_30j}")
        logger.warning(f"⚠️  Promotion refusée : {' | '.join(reasons)}")

    # 9. MLflow logging
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"retrain_{today}") as run_ctx:
        run_id = run_ctx.info.run_id

        # Paramètres
        mlflow.log_params({
            'train_start'    : TRAIN_START,
            'hold_out_start' : hold_out_start,
            'hold_out_weeks' : HOLD_OUT_WEEKS,
            'run_date'       : today,
            'dry_run'        : dry_run,
        })
        mlflow.log_params({f'7j_{k}':  v for k, v in BEST_PARAMS_7J.items()})
        mlflow.log_params({f'30j_{k}': v for k, v in BEST_PARAMS_30J.items()})

        # Métriques holdout
        mlflow.log_metrics({
            'mape_holdout_7j'  : mape_7j,
            'mae_holdout_7j'   : mae_7j,
            'mape_holdout_30j' : mape_30j,
            'mae_holdout_30j'  : mae_30j,
            'n_weeks_7j'       : len(weeks_7j),
            'n_weeks_30j'      : len(weeks_30j),
        })

        # Métriques overfitting
        mlflow.log_metrics({
            'overfit_gap_7j_pct'  : overfit_7j['gap_pct'],
            'overfit_pvalue_7j'   : overfit_7j['p_value'],
            'overfit_alert_7j'    : int(overfit_7j['overfit_alert']),
            'overfit_gap_30j_pct' : overfit_30j['gap_pct'],
            'overfit_pvalue_30j'  : overfit_30j['p_value'],
            'overfit_alert_30j'   : int(overfit_30j['overfit_alert']),
        })

        # Enregistrement dans le Model Registry
        logger.info("Enregistrement des modèles dans le Model Registry...")
        mlflow.prophet.log_model(m_7j,  artifact_path='model_7j',
                                  registered_model_name=MODEL_NAMES['7j'])
        mlflow.prophet.log_model(m_30j, artifact_path='model_30j',
                                  registered_model_name=MODEL_NAMES['30j'])

        # Promotion si applicable
        if promote and not dry_run:
            client = mlflow.tracking.MlflowClient(MLFLOW_DB)
            version_7j  = promote_to_production(client, MODEL_NAMES['7j'],  mape_7j,  run_id)
            version_30j = promote_to_production(client, MODEL_NAMES['30j'], mape_30j, run_id)
            mlflow.log_metrics({
                'promoted_version_7j' : int(version_7j),
                'promoted_version_30j': int(version_30j),
            })

        # Tags de synthèse
        mlflow.set_tags({
            'run_date'        : today,
            'hold_out_start'  : hold_out_start,
            'overfit_7j'      : str(overfit_7j['overfit_alert']),
            'overfit_30j'     : str(overfit_30j['overfit_alert']),
            'promote_decision': 'promoted' if (promote and not dry_run) else
                                'dry_run'  if (promote and dry_run)     else
                                'rejected',
            'reject_reason_7j' : '' if ok_7j  else reason_7j,
            'reject_reason_30j': '' if ok_30j else reason_30j,
            'status'           : '✅' if promote else '⚠️',
        })

        logger.info(f"MLflow run : {run_id}")

    logger.info(f"{'='*60}")
    logger.info(f"  Retrain terminé : {'PROMU ✅' if (promote and not dry_run) else 'NON PROMU ⚠️'}")
    logger.info(f"{'='*60}")

    return {
        'promoted'   : promote and not dry_run,
        'mape_7j'    : mape_7j,
        'mape_30j'   : mape_30j,
        'overfit_7j' : overfit_7j,
        'overfit_30j': overfit_30j,
    }


# CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Réentraînement avec holdout 8 semaines')
    parser.add_argument('--dry-run', action='store_true',
                        help='Évalue sans promouvoir en Production')
    args = parser.parse_args()
    run(dry_run=args.dry_run)
