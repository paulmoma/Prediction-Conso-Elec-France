"""
src/data.py
Chargement, parsing et validation des données RTE + Open-Meteo.
"""

import os
import re
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Points ruraux pondérés (hors îlots de chaleur) ───────────────────────────
POINTS_RURAUX = {
    'Alencon'    : {'lat': 48.43, 'lon':  0.08, 'poids': 0.35},
    'Bar_le_Duc' : {'lat': 48.77, 'lon':  5.16, 'poids': 0.30},
    'Perigueux'  : {'lat': 45.18, 'lon':  0.72, 'poids': 0.20},
    'Montelimar' : {'lat': 44.56, 'lon':  4.75, 'poids': 0.15},
}

DATA_DIR = Path('data')


# ════════════════════════════════════════════════════════════════════════════════
# RTE : Consommation
# ════════════════════════════════════════════════════════════════════════════════

def load_rte_xls(filepath: str) -> pd.DataFrame:
    """
    Parse un fichier RTE eco2mix (.xls en réalité TSV).

    Format attendu :
        Journée du DD/MM/YYYY
        Heures\\tPrévisionJ-1\\tPrévisionJ\\tConsommation
        00:00\\t46800\\t48000\\t46008
        ...

    Returns:
        DataFrame avec colonnes [ds (datetime), y (MW)]
    """
    rows = []
    current_date = None

    with open(filepath, 'r', encoding='windows-1252') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
            if match:
                current_date = pd.to_datetime(match.group(1), dayfirst=True)
                continue

            if line.startswith('Heures'):
                continue

            parts = line.split('\t')
            if re.match(r'^\d{2}:\d{2}$', parts[0]) and current_date is not None:
                conso = pd.to_numeric(parts[3], errors='coerce') \
                    if len(parts) > 3 else np.nan
                rows.append({
                    'ds': current_date + pd.to_timedelta(parts[0] + ':00'),
                    'y' : conso
                })

    return pd.DataFrame(rows)


def load_rte_daily(filepaths: list[str]) -> pd.DataFrame:
    """
    Charge plusieurs fichiers RTE, concatène et agrège en journalier.

    Args:
        filepaths: liste de chemins vers les fichiers .xls RTE

    Returns:
        DataFrame journalier avec colonnes [ds, y (MW moyen), total_GWh]
    """
    frames = []
    for fp in filepaths:
        logger.info(f"Chargement : {fp}")
        df_year = load_rte_xls(fp)
        logger.info(f"  → {len(df_year)} lignes | "
                    f"{df_year['ds'].min().date()} → {df_year['ds'].max().date()}")
        frames.append(df_year)

    df_15min = pd.concat(frames).dropna().sort_values('ds').reset_index(drop=True)

    # Diagnostic : jours incomplets
    counts = df_15min.set_index('ds')['y'].resample('D').count()
    jours_valides = counts[counts >= 90].index

    df_daily = (
        df_15min
        .set_index('ds')['y']
        .resample('D')
        .agg(y='mean', total_GWh=lambda x: x.sum() * 0.25 / 1000)
        .reset_index()
    )
    df_daily = df_daily[df_daily['ds'].isin(jours_valides)].reset_index(drop=True)

    # Compléter les trous (interpolation linéaire)
    date_range = pd.date_range(df_daily['ds'].min(), df_daily['ds'].max(), freq='D')
    df_daily = (
        df_daily.set_index('ds').reindex(date_range).rename_axis('ds').reset_index()
    )
    df_daily['y']         = df_daily['y'].interpolate(method='linear')
    df_daily['total_GWh'] = df_daily['total_GWh'].interpolate(method='linear')

    logger.info(f"Série journalière : {len(df_daily)} jours | "
                f"{df_daily['ds'].min().date()} → {df_daily['ds'].max().date()}")
    return df_daily


# ════════════════════════════════════════════════════════════════════════════════
# Open-Meteo : Températures
# ════════════════════════════════════════════════════════════════════════════════

def _fetch_open_meteo(lat: float, lon: float,
                      start: str, end: str,
                      timeout: int = 15) -> pd.DataFrame:
    """Appel bas niveau à l'API archive Open-Meteo."""
    r = requests.get(
        'https://archive-api.open-meteo.com/v1/archive',
        params={
            'latitude'  : lat, 'longitude': lon,
            'start_date': start, 'end_date': end,
            'daily'     : 'temperature_2m_mean,temperature_2m_min,temperature_2m_max',
            'timezone'  : 'Europe/Paris',
        },
        timeout=timeout
    ).json()

    if 'daily' not in r:
        raise ConnectionError(f"Open-Meteo archive error : {r}")

    return pd.DataFrame({
        'ds'      : pd.to_datetime(r['daily']['time']),
        'temp'    : r['daily']['temperature_2m_mean'],
        'temp_min': r['daily']['temperature_2m_min'],
        'temp_max': r['daily']['temperature_2m_max'],
    })


def _fetch_open_meteo_forecast(lat: float, lon: float,
                                horizon_days: int = 7,
                                timeout: int = 10) -> pd.DataFrame:
    """Prévisions météo Open-Meteo (jusqu'à J+16)."""
    r = requests.get(
        'https://api.open-meteo.com/v1/forecast',
        params={
            'latitude'      : lat, 'longitude': lon,
            'daily'         : 'temperature_2m_mean,temperature_2m_min,temperature_2m_max',
            'forecast_days' : min(horizon_days, 16),
            'timezone'      : 'Europe/Paris',
        },
        timeout=timeout
    ).json()

    if 'daily' not in r:
        raise ConnectionError(f"Open-Meteo forecast error : {r}")

    return pd.DataFrame({
        'ds'      : pd.to_datetime(r['daily']['time']),
        'temp'    : r['daily']['temperature_2m_mean'],
        'temp_min': r['daily']['temperature_2m_min'],
        'temp_max': r['daily']['temperature_2m_max'],
    })


def get_temperature_weighted(start: str, end: str,
                              points: dict = POINTS_RURAUX,
                              cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Température pondérée nationale à partir de N points ruraux.
    Utilise un cache CSV par point pour éviter les re-téléchargements.

    Args:
        start, end : dates ISO 'YYYY-MM-DD'
        points     : dict {nom: {lat, lon, poids}}
        cache_dir  : dossier de cache

    Returns:
        DataFrame [ds, temp, temp_min, temp_max] pondéré
    """
    cache_dir.mkdir(exist_ok=True)
    dfs = {}

    for nom, info in points.items():
        cache_file = cache_dir / f'temp_{nom}.csv'

        if cache_file.exists():
            df_cached = pd.read_csv(cache_file, parse_dates=['ds'])
            # Vérifier que le cache couvre la plage demandée
            if (df_cached['ds'].min().date() <= pd.Timestamp(start).date() and
                    df_cached['ds'].max().date() >= pd.Timestamp(end).date()):
                dfs[nom] = df_cached[
                    (df_cached['ds'] >= start) & (df_cached['ds'] <= end)
                ].reset_index(drop=True)
                logger.info(f"  {nom} → cache ✅")
                continue

        logger.info(f"  {nom} → téléchargement...")
        df_new = _fetch_open_meteo(info['lat'], info['lon'], start, end)
        df_new.to_csv(cache_file, index=False)
        dfs[nom] = df_new

    # Température pondérée
    dates     = dfs[list(points.keys())[0]]['ds']
    sum_poids = sum(v['poids'] for v in points.values())

    df_weighted = pd.DataFrame({'ds': dates})
    for col in ['temp', 'temp_min', 'temp_max']:
        df_weighted[col] = sum(
            dfs[nom][col].values * info['poids']
            for nom, info in points.items()
        ) / sum_poids

    return df_weighted


def get_temperature_forecast(horizon_days: int = 7,
                              points: dict = POINTS_RURAUX) -> pd.DataFrame:
    """
    Prévisions de température pondérées pour les N prochains jours.
    Utilisé en production pour alimenter le modèle.

    Args:
        horizon_days : nombre de jours à prévoir (max 16)
        points       : dict {nom: {lat, lon, poids}}

    Returns:
        DataFrame [ds, temp, temp_min, temp_max]
    """
    dfs = {}
    for nom, info in points.items():
        dfs[nom] = _fetch_open_meteo_forecast(info['lat'], info['lon'], horizon_days)

    dates     = dfs[list(points.keys())[0]]['ds']
    sum_poids = sum(v['poids'] for v in points.values())

    df_forecast = pd.DataFrame({'ds': dates})
    for col in ['temp', 'temp_min', 'temp_max']:
        df_forecast[col] = sum(
            dfs[nom][col].values * info['poids']
            for nom, info in points.items()
        ) / sum_poids

    return df_forecast


# ════════════════════════════════════════════════════════════════════════════════
# Validation des données : évite le bug des -5.6°C
# ════════════════════════════════════════════════════════════════════════════════

def validate_temperature(df: pd.DataFrame, name: str = 'temp') -> None:
    """
    Vérifie la cohérence d'un DataFrame de températures.
    Lève une ValueError si les données semblent corrompues.

    Checks :
        - Pas de valeurs constantes (bug de ffill sur NaN)
        - Plage physiquement plausible pour la France
        - Pas trop de NaN
    """
    issues = []

    # 1. Valeurs constantes → signe d'un ffill sur NaN
    for col in ['temp', 'temp_min', 'temp_max']:
        if col in df.columns:
            n_unique = df[col].nunique()
            if n_unique <= 3:
                issues.append(
                    f"⚠️  {col} : seulement {n_unique} valeur(s) unique(s) "
                    f"→ probablement corrompu (bug ffill ?)"
                )

    # 2. Plage physique : France métropolitaine [-20°C, +45°C]
    for col in ['temp', 'temp_min', 'temp_max']:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mn < -20 or mx > 45:
                issues.append(
                    f"⚠️  {col} : hors plage physique France [{mn:.1f}, {mx:.1f}]°C"
                )

    # 3. NaN
    n_nan = df[['temp','temp_min','temp_max']].isna().sum().sum()
    if n_nan > 0:
        issues.append(f"⚠️  {n_nan} NaN dans les températures")

    if issues:
        msg = f"\nValidation température '{name}' :\n" + '\n'.join(issues)
        raise ValueError(msg)

    logger.info(f"✅ Températures '{name}' validées : "
                f"range temp : [{df['temp'].min():.1f}, {df['temp'].max():.1f}]°C")


def build_df_model(df_daily: pd.DataFrame,
                   df_temp: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne consommation + température et valide le résultat.

    Args:
        df_daily : [ds, y, total_GWh]
        df_temp  : [ds, temp, temp_min, temp_max]

    Returns:
        DataFrame fusionné et validé
    """
    df = df_daily.merge(df_temp, on='ds', how='left')

    # Interpolation sécurisée
    for col in ['temp', 'temp_min', 'temp_max']:
        df[col] = df[col].interpolate(method='linear')

    # Validation obligatoire
    validate_temperature(df, name='df_model')

    logger.info(f"df_model : {df['ds'].min().date()} → {df['ds'].max().date()} "
                f"| {len(df)} jours | NaN y : {df['y'].isna().sum()}")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# Entrée principale pour test rapide
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Test : températures pondérées
    print("Test get_temperature_weighted...")
    df_temp = get_temperature_weighted('2025-01-01', '2025-01-07')
    validate_temperature(df_temp, 'test')
    print(df_temp)

    # Test : prévisions
    print("\nTest get_temperature_forecast...")
    df_fc = get_temperature_forecast(horizon_days=7)
    validate_temperature(df_fc, 'forecast')
    print(df_fc)
