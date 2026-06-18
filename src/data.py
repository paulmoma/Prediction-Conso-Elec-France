"""
src/data.py
Chargement, parsing et validation des données RTE + Open-Meteo.
"""

import os
import re
import base64
import logging
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Points ruraux pondérés (hors îlots de chaleur)
POINTS_RURAUX = {
    'Alencon'    : {'lat': 48.43, 'lon':  0.08, 'poids': 0.35},
    'Bar_le_Duc' : {'lat': 48.77, 'lon':  5.16, 'poids': 0.30},
    'Perigueux'  : {'lat': 45.18, 'lon':  0.72, 'poids': 0.20},
    'Montelimar' : {'lat': 44.56, 'lon':  4.75, 'poids': 0.15},
}

DATA_DIR = Path('data')


# RTE : Consommation

def load_rte_xls(filepath: str) -> pd.DataFrame:
    """Parse un fichier RTE eco2mix (.xls en réalité TSV), retourne [ds, y (MW)]."""
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
    """Charge plusieurs fichiers RTE, concatène et agrège en journalier [ds, y (MW), total_GWh]."""
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


def load_rte_complete(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Source unique de données RTE. Utilise rte_clean.csv comme data lake local :
    - Si le fichier existe et est à jour : chargement direct (< 0.1s, aucun appel API).
    - Si le fichier est en retard : appel API uniquement pour les jours manquants,
      puis mise à jour du fichier.
    - Si le fichier est absent : construction initiale depuis les XLS + API,
      puis sauvegarde. Les XLS ne sont plus re-parsés par la suite.
    Retourne [ds, y (MW moyen), total_GWh].
    """
    clean_path = data_dir / 'rte_clean.csv'
    yesterday  = date.today() - timedelta(days=1)

    if clean_path.exists():
        df_rte    = pd.read_csv(clean_path, parse_dates=['ds'])
        last_date = df_rte['ds'].max().date()

        if last_date >= yesterday:
            logger.info(f"RTE : rte_clean.csv à jour ({last_date})")
            return df_rte

        # Mise à jour incrémentale (fenêtre min 2 jours exigée par short_term API)
        start_api = str(last_date)
        logger.info(f"RTE : mise à jour {start_api} → {yesterday}")
        try:
            df_api = download_rte_daily(start=start_api, end=str(yesterday),
                                         cache_dir=data_dir)
            if not df_api.empty:
                df_rte = (pd.concat([df_rte, df_api])
                          .drop_duplicates('ds', keep='last')
                          .sort_values('ds')
                          .reset_index(drop=True))
                df_rte.to_csv(clean_path, index=False)
                logger.info(f"RTE : rte_clean.csv mis à jour → {df_rte['ds'].max().date()}")
        except Exception as e:
            logger.warning(f"API RTE indisponible, données locales utilisées : {e}")
        return df_rte

    # Premier run : construction depuis les XLS + API
    logger.info("RTE : construction initiale de rte_clean.csv depuis les XLS...")
    rte_files = sorted(data_dir.glob('conso_mix_RTE_*.xls'))
    if not rte_files:
        raise FileNotFoundError(
            "Aucun fichier conso_mix_RTE_*.xls dans data/ : "
            "télécharger depuis https://www.rte-france.com/eco2mix"
        )
    df_rte = load_rte_daily([str(f) for f in rte_files])

    last_xls_date = df_rte['ds'].max().date()
    if last_xls_date < yesterday:
        try:
            df_api = download_rte_daily(
                start=str(last_xls_date + timedelta(days=1)),
                end=str(yesterday), cache_dir=data_dir
            )
            if not df_api.empty:
                df_rte = (pd.concat([df_rte, df_api])
                          .drop_duplicates('ds', keep='last')
                          .sort_values('ds')
                          .reset_index(drop=True))
        except Exception as e:
            logger.warning(f"API RTE indisponible lors de l'initialisation : {e}")

    df_rte.to_csv(clean_path, index=False)
    logger.info(f"RTE : rte_clean.csv créé — {len(df_rte)} jours "
                f"({df_rte['ds'].min().date()} → {df_rte['ds'].max().date()})")
    return df_rte


# Open-Meteo : Températures

def _fetch_open_meteo(lat: float, lon: float,
                      start: str, end: str,
                      timeout: int = 45,
                      retries: int = 3) -> pd.DataFrame:
    """Appel bas niveau à l'API archive Open-Meteo."""
    import time
    last_err = None
    for attempt in range(retries):
        try:
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
            break
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"Open-Meteo archive timeout (tentative {attempt+1}/{retries}), retry dans {wait}s...")
                time.sleep(wait)
    else:
        raise ConnectionError(f"Open-Meteo archive inaccessible après {retries} tentatives : {last_err}")

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
    Retourne [ds, temp, temp_min, temp_max].
    """
    cache_dir.mkdir(exist_ok=True)
    dfs = {}

    for nom, info in points.items():
        cache_file = cache_dir / f'temp_{nom}.csv'

        if cache_file.exists():
            df_cached = pd.read_csv(cache_file, parse_dates=['ds'])
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


def _climatology_for_dates(dates: pd.Series,
                            cache_dir: Path = DATA_DIR,
                            points: dict = POINTS_RURAUX,
                            window: int = 15) -> pd.DataFrame:
    """
    Moyenne climatologique pondérée par jour de l'année, calculée depuis
    les fichiers cache historiques. Lissage sur une fenêtre de `window` jours
    pour éviter les artefacts de jour de l'année.
    """
    sum_poids = sum(v['poids'] for v in points.values())
    clim_by_point = {}

    for nom, info in points.items():
        cache_file = cache_dir / f'temp_{nom}.csv'
        if not cache_file.exists():
            logger.warning(f"Cache absent pour {nom}, climatologie indisponible")
            return pd.DataFrame()
        df_hist = pd.read_csv(cache_file, parse_dates=['ds'])
        df_hist['doy'] = df_hist['ds'].dt.dayofyear
        for col in ['temp', 'temp_min', 'temp_max']:
            clim_by_point[f'{nom}_{col}'] = (
                df_hist.groupby('doy')[col].mean()
                       .rolling(window, center=True, min_periods=1).mean()
            )

    result = pd.DataFrame({'ds': dates.reset_index(drop=True)})
    result['doy'] = result['ds'].dt.dayofyear
    for col in ['temp', 'temp_min', 'temp_max']:
        result[col] = sum(
            result['doy'].map(clim_by_point[f'{nom}_{col}']) * info['poids']
            for nom, info in points.items()
        ) / sum_poids

    return result[['ds', 'temp', 'temp_min', 'temp_max']]


def get_temperature_forecast(horizon_days: int = 7,
                              points: dict = POINTS_RURAUX,
                              cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Prévisions de température pondérées pour les N prochains jours.
    J+1 à J+16 : Open-Meteo Forecast API.
    J+17 à J+horizon : moyenne climatologique depuis les caches historiques.
    """
    dfs = {}
    for nom, info in points.items():
        dfs[nom] = _fetch_open_meteo_forecast(info['lat'], info['lon'],
                                               min(horizon_days, 16))

    dates     = dfs[list(points.keys())[0]]['ds']
    sum_poids = sum(v['poids'] for v in points.values())

    df_forecast = pd.DataFrame({'ds': dates})
    for col in ['temp', 'temp_min', 'temp_max']:
        df_forecast[col] = sum(
            dfs[nom][col].values * info['poids']
            for nom, info in points.items()
        ) / sum_poids
    df_forecast['source'] = 'forecast'

    if horizon_days > 16:
        last_fc_date = df_forecast['ds'].max()
        extra_dates  = pd.date_range(
            start=last_fc_date + pd.Timedelta(days=1),
            periods=horizon_days - 16,
            freq='D'
        )
        df_clim = _climatology_for_dates(pd.Series(extra_dates), cache_dir, points)
        if df_clim.empty:
            logger.warning("Climatologie indisponible, ffill utilisé pour J+17 a J+30")
            df_clim = pd.DataFrame({'ds': extra_dates})
            for col in ['temp', 'temp_min', 'temp_max']:
                df_clim[col] = df_forecast[col].iloc[-1]
            df_clim['source'] = 'ffill'
        else:
            df_clim['source'] = 'climatology'
        df_forecast = pd.concat([df_forecast, df_clim], ignore_index=True)
        logger.info(f"Température J+17 a J+{horizon_days} : moyenne climatologique")

    return df_forecast


# API RTE : Consommation temps réel

def _rte_get_token(timeout: int = 10) -> str:
    """Obtient un token OAuth2 RTE depuis RTE_CLIENT_ID / RTE_CLIENT_SECRET (.env)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    client_id     = os.environ.get('RTE_CLIENT_ID', '')
    client_secret = os.environ.get('RTE_CLIENT_SECRET', '')
    if not client_id or not client_secret:
        raise EnvironmentError(
            "Variables RTE_CLIENT_ID et RTE_CLIENT_SECRET requises "
            "(fichier .env à la racine du projet)"
        )
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        'https://digital.iservices.rte-france.com/token/oauth/',
        headers={
            'Authorization': f'Basic {credentials}',
            'Content-Type' : 'application/x-www-form-urlencoded',
        },
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()['access_token']


def _paris_offset(date_str: str) -> str:
    """Retourne le décalage UTC de Paris pour une date donnée (+01:00 CET ou +02:00 CEST)."""
    ts     = pd.Timestamp(date_str).tz_localize('Europe/Paris')
    hours  = int(ts.utcoffset().total_seconds() / 3600)
    return f"+{hours:02d}:00"


def _rte_fetch_realised(token: str, start: str, end: str,
                         timeout: int = 30) -> list:
    """
    Récupère la consommation réalisée (REALISED) via short_term RTE.
    Disponible depuis 2012, max 186 jours par appel, pas 15 min, valeurs en MW.
    start/end : dates YYYY-MM-DD (incluses).
    """
    r = requests.get(
        'https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term',
        headers={'Authorization': f'Bearer {token}'},
        params={
            'type'      : 'REALISED',
            'start_date': f'{start}T00:00:00{_paris_offset(start)}',
            'end_date'  : f'{end}T23:59:59{_paris_offset(end)}',
        },
        timeout=timeout
    )
    r.raise_for_status()
    records = []
    for serie in r.json().get('short_term', []):
        if serie.get('type') == 'REALISED':
            records.extend(serie.get('values', []))
    return records


def download_rte_daily(start: str, end: str,
                        cache_dir: Path = DATA_DIR,
                        chunk_days: int = 180) -> pd.DataFrame:
    """
    Télécharge la consommation journalière RTE via short_term/REALISED.
    Retourne [ds, y (MW moyen), total_GWh]. Met à jour rte_api_cache.csv.
    Ne télécharge que les jours absents du cache.
    Données disponibles depuis 2012, max 186 jours par appel.
    """
    cache_file   = cache_dir / 'rte_api_cache.csv'
    df_cached    = pd.DataFrame()
    actual_start = start

    if cache_file.exists():
        df_cached = pd.read_csv(cache_file, parse_dates=['ds'])
        if not df_cached.empty:
            cache_max = df_cached['ds'].max().date()
            if cache_max >= pd.Timestamp(end).date():
                logger.info(f"RTE API : cache à jour jusqu'au {cache_max}")
                return df_cached[
                    (df_cached['ds'] >= start) & (df_cached['ds'] <= end)
                ].reset_index(drop=True)
            actual_start = str(cache_max + timedelta(days=1))
            logger.info(f"RTE API : cache jusqu'au {cache_max}, "
                        f"téléchargement depuis {actual_start}")

    token = _rte_get_token()
    logger.info("Token RTE obtenu")

    all_records = []
    current = pd.Timestamp(actual_start)
    end_ts  = pd.Timestamp(end)
    while current <= end_ts:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end_ts)
        logger.info(f"  RTE API short_term/REALISED : {current.date()} → {chunk_end.date()}")
        try:
            records = _rte_fetch_realised(token, str(current.date()), str(chunk_end.date()))
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"  RTE API : échec chunk {current.date()}→{chunk_end.date()} : {e}")
        current = chunk_end + pd.Timedelta(days=1)

    if not all_records:
        logger.warning("RTE API : aucune donnée reçue pour cette période")
        return df_cached[
            (df_cached['ds'] >= start) & (df_cached['ds'] <= end)
        ].reset_index(drop=True) if not df_cached.empty else pd.DataFrame()

    # Parse 15-min MW → journalier (heure locale Paris pour éviter les bugs DST)
    rows = []
    for rec in all_records:
        ts  = pd.Timestamp(rec['start_date']).tz_convert('Europe/Paris').tz_localize(None).normalize()
        val = rec.get('value')
        if val is not None:
            rows.append({'ds': ts, 'y': float(val)})

    df_15min = pd.DataFrame(rows).dropna()
    counts   = df_15min.groupby('ds')['y'].count()
    jours_valides = counts[counts >= 90].index

    df_new = df_15min.groupby('ds').agg(
        y        =('y', 'mean'),
        total_GWh=('y', lambda x: x.sum() * 0.25 / 1000)
    ).reset_index()
    df_new = df_new[df_new['ds'].isin(jours_valides)].reset_index(drop=True)

    df_full = (pd.concat([df_cached, df_new])
               .drop_duplicates('ds', keep='last')
               .sort_values('ds')
               .reset_index(drop=True)) if not df_cached.empty else df_new

    df_full.to_csv(cache_file, index=False)
    logger.info(f"RTE API : {len(df_new)} nouveaux jours → {cache_file.name} "
                f"({len(df_full)} total)")

    return df_full[
        (df_full['ds'] >= start) & (df_full['ds'] <= end)
    ].reset_index(drop=True)


# Validation des données

def validate_temperature(df: pd.DataFrame, name: str = 'temp') -> None:
    """
    Vérifie la cohérence d'un DataFrame de températures.
    Lève une ValueError si les données semblent corrompues (valeurs constantes,
    hors plage physique France, ou NaN).
    """
    issues = []

    # Valeurs constantes → signe d'un ffill sur NaN
    for col in ['temp', 'temp_min', 'temp_max']:
        if col in df.columns:
            n_unique = df[col].nunique()
            if n_unique <= 3:
                issues.append(
                    f"⚠️  {col} : seulement {n_unique} valeur(s) unique(s) "
                    f"→ probablement corrompu (bug ffill ?)"
                )

    # Plage physique : France métropolitaine [-20°C, +45°C]
    for col in ['temp', 'temp_min', 'temp_max']:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mn < -20 or mx > 45:
                issues.append(
                    f"⚠️  {col} : hors plage physique France [{mn:.1f}, {mx:.1f}]°C"
                )

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
    """Fusionne consommation [ds, y] et température [ds, temp*], interpole les trous."""
    df = df_daily.merge(df_temp, on='ds', how='left')

    for col in ['temp', 'temp_min', 'temp_max']:
        df[col] = df[col].interpolate(method='linear')

    validate_temperature(df, name='df_model')

    logger.info(f"df_model : {df['ds'].min().date()} → {df['ds'].max().date()} "
                f"| {len(df)} jours | NaN y : {df['y'].isna().sum()}")
    return df


# Entrée principale pour test rapide
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Test get_temperature_weighted...")
    df_temp = get_temperature_weighted('2025-01-01', '2025-01-07')
    validate_temperature(df_temp, 'test')
    print(df_temp)

    print("\nTest get_temperature_forecast...")
    df_fc = get_temperature_forecast(horizon_days=7)
    validate_temperature(df_fc, 'forecast')
    print(df_fc)
