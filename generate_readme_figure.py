"""
generate_readme_figure.py
Génère figures/readme_preview.png : 5 semaines de prévisions 7j sur 2026.

Usage:
    python generate_readme_figure.py
"""

import warnings
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.data import (
    POINTS_RURAUX,
    build_df_model,
    get_temperature_weighted,
    load_rte_complete,
)
from src.features import BEST_PARAMS_7J, FEATURE_COLS_7J, make_all_features
from src.model import build_model

DATA_DIR = Path('data')
OUT_PATH = Path('figures/readme_preview.png')

SEMAINES = [
    ('2026-01-05', "Première semaine de l'année"),
    ('2026-02-16', 'Vacances hiver zone A'),
    ('2026-03-23', 'Fin mars'),
    ('2026-04-06', 'Semaine de Pâques'),
    ('2026-05-04', 'Ponts de mai'),
]

TRAIN_START = '2023-01-01'


def build_data() -> pd.DataFrame:
    df_rte  = load_rte_complete(DATA_DIR)
    df_temp = get_temperature_weighted(
        start=TRAIN_START, end=str(date.today()),
        points=POINTS_RURAUX, cache_dir=DATA_DIR
    )
    df_model = build_df_model(df_rte, df_temp)
    return make_all_features(df_model, model='7j')


def train_and_predict(df: pd.DataFrame, train_end: pd.Timestamp):
    df_train = df[
        (df['ds'] >= TRAIN_START) & (df['ds'] <= train_end)
    ][['ds', 'y'] + FEATURE_COLS_7J].dropna()

    m = build_model('7j')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(df_train)

    fut = m.make_future_dataframe(periods=7)
    fut = fut.merge(df[['ds'] + FEATURE_COLS_7J], on='ds', how='left').ffill()
    return m.predict(fut)


def make_figure(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(len(SEMAINES), 1, figsize=(13, 15))
    fig.suptitle(
        'Prévisions de consommation électrique — 7 jours\n'
        'Modèle Prophet + régresseurs météo et vacances scolaires',
        fontsize=12, y=1.01
    )

    for ax, (week_str, label) in zip(axes, SEMAINES):
        week_start = pd.Timestamp(week_str)
        week_end   = week_start + pd.Timedelta(days=6)
        train_end  = week_start - pd.Timedelta(days=1)
        ctx_start  = week_start - pd.Timedelta(days=10)

        fc       = train_and_predict(df, train_end)
        df_ctx   = df[(df['ds'] >= ctx_start) & (df['ds'] <= week_end)]
        df_pred  = fc[fc['ds'].between(week_start, week_end)]
        df_real  = df[df['ds'].between(week_start, week_end)].dropna(subset=['y'])

        mape = mean_absolute_percentage_error(df_real['y'].values,
                                              df_pred['yhat'].values) * 100
        mae  = mean_absolute_error(df_real['y'].values, df_pred['yhat'].values)

        ax.plot(df_ctx['ds'], df_ctx['y'] / 1e3,
                color='#333333', lw=1.2, ls='--', label='Réel', zorder=3)
        ax.plot(df_pred['ds'], df_pred['yhat'] / 1e3,
                color='#2e7d32', lw=2.5, label='Prévision', zorder=4)
        ax.fill_between(df_pred['ds'],
                        df_pred['yhat_lower'] / 1e3,
                        df_pred['yhat_upper'] / 1e3,
                        alpha=0.18, color='#2e7d32', zorder=2)

        pct = df_real['pct_vac'].mean() if 'pct_vac' in df_real.columns else 0

        ax.axvline(week_start, color='#888888', ls=':', lw=1.2, zorder=5)
        ax.text(0.985, 0.94,
                f'MAPE = {mape:.1f}%\nMAE  = {mae/1e3:.1f} GW',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.set_ylabel('GW', fontsize=9)
        ax.set_xlim(ctx_start, week_end + pd.Timedelta(hours=12))
        ax.grid(alpha=0.25)

        title = f'{label}   {week_str} — {week_end.date()}'
        if pct > 0.05:
            title += f'   [{pct:.0%} élèves en vacances]'
        ax.set_title(title, fontsize=9, pad=4)

        if ax is axes[0]:
            ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    OUT_PATH.parent.mkdir(exist_ok=True)
    print("Chargement des données...")
    df = build_data()
    print(f"  {len(df)} jours ({df['ds'].min().date()} → {df['ds'].max().date()})")
    print("Génération de la figure (5 entraînements)...")
    fig = make_figure(df)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f"  ✅ Sauvegardé → {OUT_PATH}")
