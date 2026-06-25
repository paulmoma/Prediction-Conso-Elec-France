"""
retrain.py
Cycle de réentraînement avec test set 8 semaines.

1. Entraîne un modèle-instrument sur [TRAIN_START → aujourd'hui - 8 semaines]
2. Évalue sur les 8 dernières semaines (MAPE + test de dérive vs validation production)
3. Si les métriques passent → réentraîne sur 100% des données, enregistre dans
   MLflow Model Registry + promeut en Production
4. Si échec → conserve le modèle Production actuel, run tagué 'rejected'
   (aucun modèle n'est enregistré dans le registry)

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

from src.data import (get_temperature_weighted, build_df_model,
                       validate_temperature, POINTS_RURAUX, load_rte_complete)
from src.features import (make_all_features, BEST_PARAMS_7J, BEST_PARAMS_30J)
from src.model import train, predict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Supprime les FutureWarnings MLflow sur les stages dépréciés (migration aliases prévue)
warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')
# Supprime les warnings de logging MLflow sur artifact_path déprécié
logging.getLogger('mlflow.models.model').setLevel(logging.ERROR)
logging.getLogger('mlflow.tracking.client').setLevel(logging.ERROR)

# Configuration
DATA_DIR        = Path('data')
MLFLOW_DB       = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'EF_Retraining'
TRAIN_START     = '2023-01-01'
TEST_SET_WEEKS  = 8

# Seuil de MAPE au-delà duquel on ne promeut pas
MAPE_MAX_PROMOTE = 5.0

# Test de dérive : la perf du run (MAPE test set de) est-elle significativement pire
# que la distribution récente des MAPE de validation observées en production ?
DRIFT_ALPHA      = 0.05   # seuil p-value Mann-Whitney
DRIFT_GAP_MAX    = 0.40   # écart relatif minimal pour déclencher l'alerte
DRIFT_MIN_REF    = 5      # nb minimal de points de référence (sinon test ignoré)
DRIFT_REF_WINDOW = 20     # nb de validations récentes servant de référence

# Noms dans le Model Registry
MODEL_NAMES = {'7j': 'prophet_7j', '30j': 'prophet_30j'}


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


def load_reference_mapes(model: str,
                         data_dir: Path = DATA_DIR,
                         window: int = DRIFT_REF_WINDOW) -> list:
    """Distribution de référence pour le test de dérive : les MAPE de validation
    production les plus récentes pour ce modèle. [] si journal absent/vide."""
    log_path = data_dir / 'validation_log.csv'
    if not log_path.exists():
        return []
    try:
        df = pd.read_csv(log_path)
        df = df[df['model'] == model].dropna(subset=['mape'])
        if df.empty:
            return []
        return df.sort_values('validation_date').tail(window)['mape'].tolist()
    except Exception as e:
        logger.warning(f"Référence dérive illisible : {e}")
        return []


def test_drift(reference_mapes: list,
               current_mapes: list,
               alpha: float = DRIFT_ALPHA,
               gap_threshold: float = DRIFT_GAP_MAX,
               min_ref: int = DRIFT_MIN_REF) -> dict:
    """Compare la MAPE du run à la distribution de référence récente (production).
    Dérive = run significativement pire (Mann-Whitney unilatéral) ET écart relatif
    > gap_threshold. Référence trop courte → test ignoré (pas d'alerte)."""
    mape_run = float(np.mean(current_mapes))

    if len(reference_mapes) < min_ref:
        logger.info(f"Dérive : référence insuffisante "
                    f"({len(reference_mapes)}/{min_ref}) → test ignoré, run={mape_run:.2f}%")
        return {'mape_ref': None, 'mape_run': round(mape_run, 3), 'gap_pct': None,
                'p_value': None, 'drift_alert': False, 'tested': False,
                'n_ref': len(reference_mapes)}

    mape_ref   = float(np.mean(reference_mapes))
    gap        = (mape_run - mape_ref) / mape_ref
    _, p_value = stats.mannwhitneyu(current_mapes, reference_mapes,
                                    alternative='greater')
    drift_alert = (p_value < alpha) and (gap > gap_threshold)

    logger.info(
        f"Dérive : réf={mape_ref:.2f}% (n={len(reference_mapes)}) run={mape_run:.2f}% "
        f"gap={gap*100:+.1f}% p={p_value:.4f} → {'DERIVE' if drift_alert else 'OK'}"
    )
    return {'mape_ref': round(mape_ref, 3), 'mape_run': round(mape_run, 3),
            'gap_pct': round(gap * 100, 1), 'p_value': round(p_value, 4),
            'drift_alert': drift_alert, 'tested': True, 'n_ref': len(reference_mapes)}


def should_promote(mape_test_set: float, drift: dict) -> tuple[bool, str]:
    """Retourne (True, raison) si le modèle peut être promu en Production."""
    if drift['drift_alert']:
        return False, (f"dérive détectée (run={drift['mape_run']:.2f}% vs "
                       f"réf={drift['mape_ref']:.2f}%, gap={drift['gap_pct']:+.1f}%, "
                       f"p={drift['p_value']:.4f})")
    if mape_test_set > MAPE_MAX_PROMOTE:
        return False, f"MAPE test set {mape_test_set:.2f}% > seuil {MAPE_MAX_PROMOTE}%"
    suffix = "" if drift['tested'] else " (réf. dérive en constitution)"
    return True, f"MAPE={mape_test_set:.2f}% OK, pas de dérive{suffix}"


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
        description=f"MAPE test set 8s={mape:.2f}% | run={run_id[:8]} | {date.today()}"
    )
    client.set_model_version_tag(model_name, new_version, 'stage', 'Production')
    client.set_model_version_tag(model_name, new_version, 'promoted_date', str(date.today()))
    client.set_model_version_tag(model_name, new_version, 'mape_test_set', f'{mape:.2f}%')
    logger.info(f"[{model_name}] v{new_version} → Production ✅")
    return new_version


# Pipeline de réentraînement

def _build_test_set_temps(df_temp_hist: pd.DataFrame,
                          test_set_start: str,
                          data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Températures pour l'évaluation test set : réelles jusqu'à test_set_start,
    puis prévisions Open-Meteo telles qu'enregistrées à cette date dans temp_forecast_log.csv.
    Fallback sur les températures réelles si le journal est absent ou trop court.
    """
    log_path = data_dir / 'temp_forecast_log.csv'
    hold_ts  = pd.Timestamp(test_set_start)

    if not log_path.exists():
        logger.warning("Test set : temp_forecast_log.csv absent → températures réelles utilisées")
        return df_temp_hist

    df_log    = pd.read_csv(log_path, parse_dates=['ds', 'run_date'])
    past_runs = df_log[df_log['run_date'].dt.normalize() <= hold_ts]['run_date'].unique()

    if len(past_runs) == 0:
        logger.warning("Test set : aucun run avant test_set_start → températures réelles utilisées")
        return df_temp_hist

    closest_run = pd.Timestamp(max(past_runs))
    df_fc = (df_log[df_log['run_date'].dt.normalize() == closest_run.normalize()]
             [['ds', 'temp', 'temp_min', 'temp_max']]
             .query('ds >= @hold_ts')
             .copy())

    df_before  = df_temp_hist[df_temp_hist['ds'] < hold_ts]
    df_after   = df_temp_hist[df_temp_hist['ds'] >= hold_ts]

    # Prévisions là où disponibles, températures réelles en fallback
    merged = df_after.merge(df_fc, on='ds', how='left', suffixes=('_real', '_fc'))
    df_test_set = merged.assign(
        temp     = lambda d: d['temp_fc'].fillna(d['temp_real']),
        temp_min = lambda d: d['temp_min_fc'].fillna(d['temp_min_real']),
        temp_max = lambda d: d['temp_max_fc'].fillna(d['temp_max_real']),
    )[['ds', 'temp', 'temp_min', 'temp_max']]

    n_fc = merged['temp_fc'].notna().sum()
    logger.info(f"Test set : prévisions du run {closest_run.date()} "
                f"({n_fc}/{len(df_after)} jours couverts, reste en réel)")

    return pd.concat([df_before, df_test_set]).reset_index(drop=True)


def run(dry_run: bool = False):
    today          = str(date.today())
    test_set_start = str(date.today() - timedelta(weeks=TEST_SET_WEEKS))
    logger.info(f"{'='*60}")
    logger.info(f"  Retrain {today} | train→{test_set_start} | test set {test_set_start}→{today}")
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

    # 3. Fusion + feature engineering sur températures réelles (pour le modèle Production)
    df_model = build_df_model(df_daily, df_temp_hist)
    df_7j    = make_all_features(df_model, model='7j')
    df_30j   = make_all_features(df_model, model='30j')

    # 3b. Features pour l'instrument d'évaluation : températures prévues sur le test set
    #     (simule les conditions réelles de production, où les températures futures ne sont
    #     pas encore connues au moment du run)
    logger.info("Construction des features d'évaluation avec températures prévues sur le test set...")
    df_temp_eval  = _build_test_set_temps(df_temp_hist, test_set_start, DATA_DIR)
    df_model_eval = build_df_model(df_daily, df_temp_eval)
    df_7j_eval    = make_all_features(df_model_eval, model='7j')
    df_30j_eval   = make_all_features(df_model_eval, model='30j')

    # 4. Entraînement du modèle-INSTRUMENT sur [TRAIN_START → test_set_start].
    #    Il sert UNIQUEMENT à mesurer la qualité sur le test set ; il n'est jamais
    #    déployé (cf. étape 9 : le modèle promu est réentraîné sur 100% des données).
    logger.info(f"Entraînement (instrument d'éval) sur [TRAIN_START → {test_set_start}]...")
    m_7j  = train(df_7j_eval,  model='7j',  train_start=TRAIN_START, train_end=test_set_start)
    m_30j = train(df_30j_eval, model='30j', train_start=TRAIN_START, train_end=test_set_start)

    # 5. Prévisions sur le test set (8 semaines = 56 jours)
    logger.info("Prévisions sur la période test set...")
    fc_7j  = predict(m_7j,  df_7j_eval,  model='7j',  horizon=TEST_SET_WEEKS * 7)
    fc_30j = predict(m_30j, df_30j_eval, model='30j', horizon=TEST_SET_WEEKS * 7)

    # 6. Évaluation semaine par semaine (y réel RTE dans df_7j_eval, inchangé)
    logger.info("Évaluation test set...")
    weeks_7j  = rolling_weekly_mape(df_7j_eval,  fc_7j,  test_set_start, today)
    weeks_30j = rolling_weekly_mape(df_30j_eval, fc_30j, test_set_start, today)

    if not weeks_7j or not weeks_30j:
        raise RuntimeError(
            "Pas assez de semaines complètes dans le test set pour évaluer. "
            "Vérifier les données RTE."
        )

    mape_7j  = np.mean([w['mape'] for w in weeks_7j])
    mape_30j = np.mean([w['mape'] for w in weeks_30j])
    mae_7j   = np.mean([w['mae']  for w in weeks_7j])
    mae_30j  = np.mean([w['mae']  for w in weeks_30j])

    logger.info(f"MAPE test set 7j  : {mape_7j:.2f}%  MAE={mae_7j:,.0f}MW  ({len(weeks_7j)} semaines)")
    logger.info(f"MAPE test set 30j : {mape_30j:.2f}%  MAE={mae_30j:,.0f}MW  ({len(weeks_30j)} semaines)")

    # 7. Test de dérive vs distribution récente des MAPE de validation production
    drift_7j  = test_drift(load_reference_mapes('7j'),  [w['mape'] for w in weeks_7j])
    drift_30j = test_drift(load_reference_mapes('30j'), [w['mape'] for w in weeks_30j])

    # 8. Décision de promotion (les deux modèles ensemble ou aucun)
    ok_7j,  reason_7j  = should_promote(mape_7j,  drift_7j)
    ok_30j, reason_30j = should_promote(mape_30j, drift_30j)
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
            'test_set_start' : test_set_start,
            'test_set_weeks' : TEST_SET_WEEKS,
            'run_date'       : today,
            'dry_run'        : dry_run,
        })
        mlflow.log_params({f'7j_{k}':  v for k, v in BEST_PARAMS_7J.items()})
        mlflow.log_params({f'30j_{k}': v for k, v in BEST_PARAMS_30J.items()})

        # Métriques test set
        mlflow.log_metrics({
            'mape_test_set_7j'  : mape_7j,
            'mae_test_set_7j'   : mae_7j,
            'mape_test_set_30j' : mape_30j,
            'mae_test_set_30j'  : mae_30j,
            'n_weeks_7j'       : len(weeks_7j),
            'n_weeks_30j'      : len(weeks_30j),
        })

        # Métriques dérive (gap/p-value uniquement quand le test a réellement tourné)
        drift_metrics = {}
        for tag, d in [('7j', drift_7j), ('30j', drift_30j)]:
            drift_metrics[f'drift_alert_{tag}']  = int(d['drift_alert'])
            drift_metrics[f'drift_tested_{tag}'] = int(d['tested'])
            if d['tested']:
                drift_metrics[f'drift_gap_{tag}_pct'] = d['gap_pct']
                drift_metrics[f'drift_pvalue_{tag}']  = d['p_value']
        mlflow.log_metrics(drift_metrics)

        # Enregistrement + promotion.
        # IMPORTANT : on déploie un modèle réentraîné sur 100% des données (train_end=None).
        # Les m_7j/m_30j ci-dessus servent UNIQUEMENT à estimer la qualité sur le test set ;
        # le modèle mis en Production, lui, doit avoir vu les 8 dernières semaines.
        # La MAPE test set reste l'estimateur (conservateur) de sa performance réelle.
        # Conséquence : un run rejeté ou en dry-run n'enregistre RIEN dans le registry,
        # et le modèle Production en place reste inchangé.
        if promote and not dry_run:
            logger.info("Réentraînement sur l'ensemble des données avant déploiement...")
            m_7j_full  = train(df_7j,  model='7j',  train_start=TRAIN_START)   # train_end=None
            m_30j_full = train(df_30j, model='30j', train_start=TRAIN_START)

            logger.info("Enregistrement des modèles dans le Model Registry...")
            mlflow.prophet.log_model(m_7j_full,  artifact_path='model_7j',
                                      registered_model_name=MODEL_NAMES['7j'])
            mlflow.prophet.log_model(m_30j_full, artifact_path='model_30j',
                                      registered_model_name=MODEL_NAMES['30j'])

            client = mlflow.tracking.MlflowClient(MLFLOW_DB)
            version_7j  = promote_to_production(client, MODEL_NAMES['7j'],  mape_7j,  run_id)
            version_30j = promote_to_production(client, MODEL_NAMES['30j'], mape_30j, run_id)
            mlflow.log_metrics({
                'promoted_version_7j' : int(version_7j),
                'promoted_version_30j': int(version_30j),
            })
        else:
            logger.info("Pas de promotion (rejet ou dry-run) : "
                        "aucun modèle enregistré dans le registry")

        # Tags de synthèse
        mlflow.set_tags({
            'run_date'        : today,
            'test_set_start'  : test_set_start,
            'drift_7j'        : str(drift_7j['drift_alert']),
            'drift_30j'       : str(drift_30j['drift_alert']),
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
        'drift_7j'   : drift_7j,
        'drift_30j'  : drift_30j,
    }


# CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Réentraînement avec test set 8 semaines')
    parser.add_argument('--dry-run', action='store_true',
                        help='Évalue sans promouvoir en Production')
    args = parser.parse_args()
    run(dry_run=args.dry_run)
