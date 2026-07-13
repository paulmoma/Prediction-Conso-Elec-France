"""
src/evaluate.py
Métriques, validation et visualisation des prévisions.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)


# Métriques

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    horizon_name: str = '') -> dict:
    """Calcule MAPE, MAE et RMSE. Retourne dict {mape, mae, rmse}."""
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    label = f" [{horizon_name}]" if horizon_name else ""
    logger.info(f"Métriques{label} : MAPE={mape:.2f}%  MAE={mae:,.0f}MW  RMSE={rmse:,.0f}MW")

    return {'mape': round(mape, 4), 'mae': round(mae, 1), 'rmse': round(rmse, 1)}


def compute_metrics_by_day(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            dates: np.ndarray) -> pd.DataFrame:
    """MAPE et MAE jour par jour. Utile pour identifier les pics d'erreur."""
    return pd.DataFrame({
        'ds'       : dates,
        'y_true'   : y_true,
        'y_pred'   : y_pred,
        'error_pct': np.abs(y_pred - y_true) / y_true * 100,
        'error_mw' : np.abs(y_pred - y_true),
    })


def naive_seasonal_forecast(df: pd.DataFrame,
                              horizon: int,
                              lag_days: int = 364) -> np.ndarray:
    """Modèle naïf saisonnier : répète les valeurs d'il y a lag_days jours (baseline)."""
    y_all  = df['y'].values
    offset = len(y_all) - horizon - lag_days
    if offset < 0:
        raise ValueError(f"Pas assez d'historique pour naïf saisonnier "
                         f"({lag_days} jours requis)")
    return y_all[offset: offset + horizon]


# Plots

def plot_forecast(df: pd.DataFrame,
                  fc: pd.DataFrame,
                  horizon: int,
                  title: str = 'Prévision Prophet',
                  context_days: int = 60) -> plt.Figure:
    """Graphique prévision vs réel avec intervalle de confiance."""
    fc_horizon = fc.tail(horizon)
    test_dates  = fc_horizon['ds'].values
    pred        = fc_horizon['yhat'].values

    y_real = df[df['ds'].isin(fc_horizon['ds'])]['y'].values

    last_train = df[df['y'].notna()]['ds'].max()
    df_ctx = df[
        (df['ds'] >= last_train - pd.Timedelta(days=context_days)) &
        (df['ds'] <= last_train)
    ]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df_ctx['ds'], df_ctx['y'] / 1e3,
            color='steelblue', lw=0.9, label='Historique')

    if len(y_real) == horizon:
        ax.plot(test_dates, y_real / 1e3,
                color='black', lw=1.5, ls='--', label='Réel')

    ax.plot(test_dates, pred / 1e3,
            color='seagreen', lw=2, label='Prophet')

    ax.fill_between(
        test_dates,
        fc_horizon['yhat_lower'].values / 1e3,
        fc_horizon['yhat_upper'].values / 1e3,
        alpha=0.2, color='seagreen', label='IC 80%'
    )

    ax.axvline(pd.Timestamp(last_train), color='gray', ls=':', lw=1.5)
    ax.set_ylabel('GW')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.tight_layout()

    return fig


def plot_error_by_day(metrics_df: pd.DataFrame,
                       mape_mean: float,
                       title: str = 'Erreur par jour') -> plt.Figure:
    """Barplot des erreurs jour par jour."""
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = ['tomato' if e > mape_mean * 1.5 else 'salmon'
              for e in metrics_df['error_pct']]

    ax.bar(range(len(metrics_df)), metrics_df['error_pct'],
           color=colors, alpha=0.8)
    ax.axhline(mape_mean, color='black', ls='--', lw=1.5,
               label=f'MAPE moy = {mape_mean:.1f}%')

    ax.set_xlabel('Jour J+')
    ax.set_ylabel('Erreur (%)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig


def plot_comparison(results: dict,
                    title: str = 'Comparaison modèles') -> plt.Figure:
    """Barplot MAPE par modèle. results = {nom: mape_value}."""
    fig, ax = plt.subplots(figsize=(8, 4))

    names  = list(results.keys())
    mapes  = list(results.values())
    colors = ['seagreen' if 'Prophet' in n else 'steelblue'
              if 'Naïf' not in n else 'tomato'
              for n in names]

    bars = ax.barh(names, mapes, color=colors, alpha=0.85)

    for bar, val in zip(bars, mapes):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=9)

    ax.set_xlabel('MAPE (%)')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    return fig


# Rapport de validation

def validation_report(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       dates: np.ndarray,
                       y_naive: np.ndarray = None,
                       horizon_name: str = '') -> dict:
    """Rapport complet : métriques + comparaison naïf + jours outliers (>2×MAPE)."""
    metrics = compute_metrics(y_true, y_pred, horizon_name)
    report  = {'metrics': metrics, 'horizon': horizon_name}

    if y_naive is not None:
        naive_metrics = compute_metrics(y_true, y_naive, f'{horizon_name}_naive')
        report['beat_naive']   = metrics['mape'] < naive_metrics['mape']
        report['naive_mape']   = naive_metrics['mape']
        report['gain_vs_naive'] = round(naive_metrics['mape'] - metrics['mape'], 3)

    daily = compute_metrics_by_day(y_true, y_pred, dates)
    threshold = metrics['mape'] * 2
    worst = daily[daily['error_pct'] > threshold][['ds', 'error_pct', 'error_mw']]
    report['worst_days'] = worst.to_dict(orient='records')
    report['n_outlier_days'] = len(worst)

    logger.info(f"{'='*45}")
    logger.info(f"Rapport validation {horizon_name}")
    logger.info(f"  MAPE  : {metrics['mape']:.2f}%")
    logger.info(f"  MAE   : {metrics['mae']:,.0f} MW")
    if y_naive is not None:
        status = '✅' if report['beat_naive'] else '❌'
        logger.info(f"  Naïf  : {naive_metrics['mape']:.2f}% {status}")
    logger.info(f"  Jours outliers (>2×MAPE) : {len(worst)}")
    logger.info(f"{'='*45}")

    return report


# Entrée principale pour test rapide
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n = 30
    y_true  = 50000 + np.random.randn(n) * 3000
    y_pred  = y_true + np.random.randn(n) * 1500
    y_naive = y_true + np.random.randn(n) * 4000
    dates   = pd.date_range('2026-01-01', periods=n).values

    report = validation_report(y_true, y_pred, dates, y_naive, '30j')
    print(f"\nBeat naïf : {report['beat_naive']}")
    print(f"Gain      : +{report['gain_vs_naive']:.2f}%")

    fig = plot_comparison({
        'Prophet 30j': report['metrics']['mape'],
        'Naïf saisonnier': report['naive_mape'],
    })
    plt.show()
    print("✅ evaluate.py OK")
