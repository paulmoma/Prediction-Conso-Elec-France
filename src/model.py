"""
src/model.py
Construction, entraînement et prédiction Prophet.
Supporte deux modèles : 7j et 30j.
"""

import logging
import warnings
from datetime import date
import pandas as pd
import numpy as np
from prophet import Prophet

from src.features import (
    BEST_PARAMS_7J, BEST_PARAMS_30J,
    FEATURE_COLS_7J, FEATURE_COLS_30J,
    get_feature_cols,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_START = '2023-01-01'


def build_model(model: str = '30j') -> Prophet:
    """
    Instancie Prophet selon le modèle cible ('7j' ou '30j').

    Args:
        model : '7j' ou '30j'

    Returns:
        Prophet non entraîné
    """
    params = BEST_PARAMS_7J if model == '7j' else BEST_PARAMS_30J

    m = Prophet(
        yearly_seasonality      = False,
        weekly_seasonality      = False,
        daily_seasonality       = False,
        changepoint_prior_scale = params['changepoint_prior_scale'],
    )
    m.add_seasonality(
        name='yearly', period=365.25,
        fourier_order=params['fourier_yearly']
    )
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    m.add_country_holidays(country_name='FR')

    # Régresseurs communs aux deux modèles
    for reg, ps_key in [
        ('HDD',       'prior_hdd_min'),
        ('CDD',       'prior_cdd_max'),
        ('HDD_mean',  'prior_hdd_mean'),
        ('CDD_mean',  'prior_cdd_mean'),
        ('lag_hiver', 'prior_lag'),
        ('lag_ete',   'prior_lag'),
    ]:
        m.add_regressor(reg, prior_scale=params[ps_key])

    # Régresseur vacances : uniquement pour le modèle 7j
    if model == '7j':
        m.add_regressor('pct_vac', prior_scale=params['prior_vac'])

    return m


def train(df: pd.DataFrame,
          model: str = '30j',
          train_start: str = DEFAULT_TRAIN_START,
          train_end: str = None) -> Prophet:
    """
    Entraîne Prophet sur la fenêtre spécifiée.

    Args:
        df          : DataFrame avec [ds, y] + features
        model       : '7j' ou '30j'
        train_start : début de la fenêtre (défaut 2023-01-01)
        train_end   : fin (None = toutes les données disponibles)

    Returns:
        Prophet entraîné
    """
    feat_cols = get_feature_cols(model)

    mask = df['ds'] >= train_start
    if train_end:
        mask &= df['ds'] <= train_end
    df_train = df[mask][['ds', 'y'] + feat_cols].dropna()

    logger.info(f"[{model}] Entraînement : "
                f"{df_train['ds'].min().date()} → "
                f"{df_train['ds'].max().date()} "
                f"({len(df_train)} jours)")

    m = build_model(model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(df_train)

    logger.info(f"[{model}] ✅ Modèle entraîné")
    return m


def predict(m: Prophet,
            df: pd.DataFrame,
            model: str = '30j',
            horizon: int = None) -> pd.DataFrame:
    """
    Génère des prévisions.

    Args:
        m       : Prophet entraîné
        df      : DataFrame complet incluant features futures
        model   : '7j' ou '30j'
        horizon : None = utilise le défaut du modèle (7 ou 30)

    Returns:
        DataFrame Prophet complet [ds, yhat, yhat_lower, yhat_upper, ...]
    """
    if horizon is None:
        horizon = 7 if model == '7j' else 30

    feat_cols = get_feature_cols(model)

    # Étendre jusqu'à aujourd'hui + horizon (le modèle peut finir avant aujourd'hui)
    last_trained = m.history['ds'].max()
    today_ts     = pd.Timestamp(date.today())
    gap          = max(0, (today_ts - last_trained).days)
    periods      = gap + horizon

    future = m.make_future_dataframe(periods=periods)
    future = future.merge(
        df[['ds'] + feat_cols], on='ds', how='left'
    ).ffill()

    fc = m.predict(future)
    logger.info(f"[{model}] Prévision : "
                f"{fc['ds'].iloc[-horizon].date()} → "
                f"{fc['ds'].iloc[-1].date()}")
    return fc


def forecast(df: pd.DataFrame,
             model: str = '30j',
             train_start: str = DEFAULT_TRAIN_START) -> pd.DataFrame:
    """
    Pipeline complète : entraîne sur tout df et retourne les N prochains jours.

    Utilisé en production : df doit inclure les features futures
    (températures prévues + pct_vacances).

    Args:
        df          : DataFrame complet [ds, y, features]
                      Les jours futurs n'ont pas de 'y'
        model       : '7j' ou '30j'
        train_start : début de la fenêtre glissante

    Returns:
        DataFrame [ds, yhat, yhat_lower, yhat_upper] pour l'horizon
    """
    horizon  = 7 if model == '7j' else 30
    m        = train(df, model=model, train_start=train_start)
    fc       = predict(m, df, model=model, horizon=horizon)

    last_known = df[df['y'].notna()]['ds'].max()
    return fc[fc['ds'] > last_known][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].reset_index(drop=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("7j  regresseurs :", FEATURE_COLS_7J)
    print("30j regresseurs :", FEATURE_COLS_30J)
    print("✅ model.py OK")
