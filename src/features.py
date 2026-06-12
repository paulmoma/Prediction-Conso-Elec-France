"""
src/features.py
Feature engineering température + vacances scolaires → régresseurs Prophet.

Deux modèles :
    7j  : BEST_PARAMS_7J  + pct_vacances
    30j : BEST_PARAMS_30J (sans vacances)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paramètres 30j (Optuna, horizon='30 days') ───────────────────────────────
BEST_PARAMS_30J = {
    'changepoint_prior_scale': 0.01,
    'fourier_yearly'  : 7,
    'heat_base_min'   : 7,
    'cool_base_max'   : 30,
    'heat_base_mean'  : 15,
    'cool_base_mean'  : 18,
    'prior_hdd_min'   : 0.4,
    'prior_cdd_max'   : 1.7,
    'prior_hdd_mean'  : 1.0,
    'prior_cdd_mean'  : 2.5,
    'prior_lag'       : 1.0,
    'lag_hiver'       : 4,
    'lag_ete'         : 1,
}

# ── Paramètres 7j (Optuna joint avec pct_vacances, horizon='7 days') ─────────
BEST_PARAMS_7J = {
    'changepoint_prior_scale': 0.2,
    'fourier_yearly'  : 11,
    'heat_base_min'   : 4,
    'cool_base_max'   : 32,
    'heat_base_mean'  : 15,
    'cool_base_mean'  : 19,
    'prior_hdd_min'   : 2.4,
    'prior_cdd_max'   : 2.9,
    'prior_hdd_mean'  : 1.5,
    'prior_cdd_mean'  : 1.5,
    'prior_lag'       : 1.0,
    'prior_vac'       : 1.3,
    'lag_hiver'       : 4,
    'lag_ete'         : 2,
}

FEATURE_COLS_30J = ['HDD', 'CDD', 'HDD_mean', 'CDD_mean', 'lag_hiver', 'lag_ete']
FEATURE_COLS_7J  = ['HDD', 'CDD', 'HDD_mean', 'CDD_mean', 'lag_hiver', 'lag_ete',
                    'pct_vac']

# ════════════════════════════════════════════════════════════════════════════════
# Vacances scolaires pondérées
# ════════════════════════════════════════════════════════════════════════════════

POIDS_ZONES = {'A': 0.32, 'B': 0.40, 'C': 0.28}

VACANCES = {
    'A': [
        ('2023-02-11','2023-02-26'), ('2023-04-22','2023-05-08'),
        ('2023-07-08','2023-09-04'), ('2023-10-21','2023-11-05'),
        ('2023-12-23','2024-01-07'), ('2024-02-10','2024-02-25'),
        ('2024-04-20','2024-05-06'), ('2024-07-06','2024-09-01'),
        ('2024-10-19','2024-11-04'), ('2024-12-21','2025-01-05'),
        ('2025-02-15','2025-03-02'), ('2025-04-19','2025-05-05'),
        ('2025-07-05','2025-09-01'), ('2025-10-18','2025-11-03'),
        ('2025-12-20','2026-01-04'), ('2026-02-14','2026-03-01'),
        ('2026-04-18','2026-05-04'),
    ],
    'B': [
        ('2023-02-04','2023-02-19'), ('2023-04-15','2023-05-02'),
        ('2023-07-08','2023-09-04'), ('2023-10-21','2023-11-05'),
        ('2023-12-23','2024-01-07'), ('2024-02-17','2024-03-03'),
        ('2024-04-27','2024-05-13'), ('2024-07-06','2024-09-01'),
        ('2024-10-19','2024-11-04'), ('2024-12-21','2025-01-05'),
        ('2025-02-22','2025-03-09'), ('2025-04-26','2025-05-12'),
        ('2025-07-05','2025-09-01'), ('2025-10-18','2025-11-03'),
        ('2025-12-20','2026-01-04'), ('2026-02-21','2026-03-08'),
        ('2026-04-25','2026-05-11'),
    ],
    'C': [
        ('2023-02-18','2023-03-05'), ('2023-04-29','2023-05-15'),
        ('2023-07-08','2023-09-04'), ('2023-10-21','2023-11-05'),
        ('2023-12-23','2024-01-07'), ('2024-02-24','2024-03-10'),
        ('2024-05-04','2024-05-20'), ('2024-07-06','2024-09-01'),
        ('2024-10-19','2024-11-04'), ('2024-12-21','2025-01-05'),
        ('2025-03-01','2025-03-16'), ('2025-05-03','2025-05-19'),
        ('2025-07-05','2025-09-01'), ('2025-10-18','2025-11-03'),
        ('2025-12-20','2026-01-04'), ('2026-02-28','2026-03-15'),
        ('2026-05-02','2026-05-18'),
    ],
}


def make_pct_vacances(dates: pd.Series) -> np.ndarray:
    """
    % d'élèves en vacances pour chaque date (0.0 → 1.0).
    Connu à l'avance → information parfaite pour les 7 prochains jours.
    """
    result = np.zeros(len(dates))
    for zone, periodes in VACANCES.items():
        w = POIDS_ZONES[zone]
        for start, end in periodes:
            mask = (dates >= start) & (dates <= end)
            result[mask.values] += w
    return result.round(4)


# ════════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ════════════════════════════════════════════════════════════════════════════════

def make_hdd_cdd(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """4 régresseurs température (non-linéarité via clip)."""
    df = df.copy()
    df['HDD']      = (params['heat_base_min']  - df['temp_min']).clip(lower=0)
    df['CDD']      = (df['temp_max'] - params['cool_base_max'] ).clip(lower=0)
    df['HDD_mean'] = (params['heat_base_mean'] - df['temp']    ).clip(lower=0)
    df['CDD_mean'] = (df['temp'] - params['cool_base_mean']    ).clip(lower=0)
    return df


def make_seasonal_lags(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """2 régresseurs de lag saisonnier."""
    df       = df.copy()
    is_hiver = df['ds'].dt.month.isin([10, 11, 12, 1, 2, 3, 4])
    lag_h    = df['temp'].shift(params['lag_hiver']).fillna(df['temp'])
    lag_e    = df['temp'].shift(params['lag_ete']  ).fillna(df['temp'])
    df['lag_hiver'] = np.where(is_hiver,  lag_h, 0)
    df['lag_ete']   = np.where(~is_hiver, lag_e, 0)
    return df


def make_all_features(df: pd.DataFrame,
                      model: str = '30j') -> pd.DataFrame:
    """
    Feature engineering complet selon le modèle cible.

    Args:
        df    : DataFrame avec [ds, temp, temp_min, temp_max]
        model : '7j' ou '30j'

    Returns:
        df enrichi avec toutes les features du modèle choisi
    """
    params = BEST_PARAMS_7J if model == '7j' else BEST_PARAMS_30J
    df     = make_hdd_cdd(df, params)
    df     = make_seasonal_lags(df, params)
    if model == '7j':
        df['pct_vac'] = make_pct_vacances(df['ds'])
    return df


def get_feature_cols(model: str = '30j') -> list:
    """Retourne la liste des colonnes features pour le modèle choisi."""
    return FEATURE_COLS_7J if model == '7j' else FEATURE_COLS_30J


if __name__ == '__main__':
    dates = pd.date_range('2026-01-01', periods=10, freq='D')
    pct   = make_pct_vacances(pd.Series(dates))
    print(pd.DataFrame({'ds': dates, 'pct_vac': pct}))
