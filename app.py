"""
app.py
Dashboard Streamlit, Prévision consommation électrique France.

Lancement : streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path

DATA_DIR     = Path('data')
FORECAST_DIR = DATA_DIR / 'forecasts'

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = '⚡ Prévision Conso France',
    page_icon   = '⚡',
    layout      = 'wide',
)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_forecast(model: str) -> pd.DataFrame:
    path = DATA_DIR / f'forecast_{model}_latest.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['ds'])
    df[['yhat', 'yhat_lower', 'yhat_upper']] /= 1e3
    return df


@st.cache_data(ttl=3600)
def load_rte_actual(days: int = 180) -> pd.DataFrame:
    """Charge le réalisé RTE depuis les .xls (fallback sur rte_daily*.csv)."""
    # Essai CSV pré-calculé
    csv_paths = sorted(DATA_DIR.glob('rte_daily*.csv'))
    if csv_paths:
        df = pd.concat([pd.read_csv(p, parse_dates=['ds']) for p in csv_paths])
        df = df.sort_values('ds').drop_duplicates('ds')
        df['y'] /= 1e3
        cutoff = pd.Timestamp(date.today() - timedelta(days=days))
        return df[df['ds'] >= cutoff].copy()

    # Fallback : lecture directe des .xls
    xls_paths = sorted(DATA_DIR.glob('conso_mix_RTE_*.xls'))
    if not xls_paths:
        return pd.DataFrame()
    try:
        from src.data import load_rte_daily
        df = load_rte_daily([str(p) for p in xls_paths])
        df['y'] /= 1e3
        cutoff = pd.Timestamp(date.today() - timedelta(days=days))
        return df[df['ds'] >= cutoff].copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_past_forecasts(model: str) -> dict[str, pd.DataFrame]:
    """Retourne {run_date: DataFrame} pour toutes les prévisions datées."""
    result = {}
    for f in sorted(FORECAST_DIR.glob(f'forecast_{model}_*.csv')):
        run_date = f.stem.split('_')[-1]
        df = pd.read_csv(f, parse_dates=['ds'])
        df[['yhat', 'yhat_lower', 'yhat_upper']] /= 1e3
        result[run_date] = df
    return result


@st.cache_data(ttl=3600)
def load_validation_log() -> pd.DataFrame:
    path = DATA_DIR / 'validation_log.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['validation_date', 'run_date'])


@st.cache_data(ttl=3600)
def load_temp_forecast() -> pd.DataFrame:
    path = DATA_DIR / 'temperature_forecast.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['ds'])


# ── Chart helpers ─────────────────────────────────────────────────────────────
def make_forecast_chart(df_hist: pd.DataFrame,
                         df_fc: pd.DataFrame,
                         title: str,
                         color: str = '#2ecc71') -> go.Figure:
    fig = go.Figure()

    if not df_hist.empty:
        fig.add_trace(go.Scatter(
            x=df_hist['ds'], y=df_hist['y'],
            name='Historique', line=dict(color='#3498db', width=1.5),
            mode='lines'
        ))

    if df_fc.empty:
        st.warning("Aucune prévision disponible, lance d'abord run_weekly.py")
        return fig

    fig.add_trace(go.Scatter(
        x=pd.concat([df_fc['ds'], df_fc['ds'][::-1]]),
        y=pd.concat([df_fc['yhat_upper'], df_fc['yhat_lower'][::-1]]),
        fill='toself', fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba'),
        line=dict(color='rgba(0,0,0,0)'),
        name='IC 80%', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=df_fc['ds'], y=df_fc['yhat'],
        name='Prévision', line=dict(color=color, width=2.5),
        mode='lines'
    ))
    fig.add_vline(x=df_fc['ds'].min(), line_dash='dot', line_color='gray', opacity=0.6)
    fig.update_layout(
        title=title, yaxis_title='GW',
        hovermode='x unified', height=350,
        legend=dict(orientation='h', y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


PALETTE = [
    '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
    '#1abc9c', '#9b59b6', '#e91e63', '#ff5722',
]


def make_validation_chart(df_rte: pd.DataFrame,
                           past_fcs: dict,
                           selected_runs: list[str],
                           model: str) -> go.Figure:
    fig = go.Figure()

    if not df_rte.empty:
        fig.add_trace(go.Scatter(
            x=df_rte['ds'], y=df_rte['y'],
            name='Réalisé RTE', line=dict(color='#3498db', width=2),
            mode='lines'
        ))

    for i, run_date in enumerate(selected_runs):
        df_fc = past_fcs.get(run_date)
        if df_fc is None:
            continue
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=df_fc['ds'], y=df_fc['yhat'],
            name=f'Prév. {run_date}',
            line=dict(color=color, width=1.5, dash='dash'),
            mode='lines'
        ))

    label = '7 jours' if model == '7j' else '30 jours'
    fig.update_layout(
        title=f'Réalisé vs Prévisions passées, modèle {label}',
        yaxis_title='GW',
        hovermode='x unified', height=420,
        legend=dict(orientation='h', y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Page principale
# ════════════════════════════════════════════════════════════════════════════════

st.title('⚡ Prévision consommation électrique journalière, périmètre France')
st.caption(f"Mise à jour : {date.today().strftime('%d %B %Y')} | "
           f"Modèle : Prophet + HDD/CDD + vacances scolaires | "
           f"MAPE prod : **2.90%** (7j)  **2.53%** (30j)")

fc_7j   = load_forecast('7j')
fc_30j  = load_forecast('30j')
df_hist = load_rte_actual(days=60)
df_temp = load_temp_forecast()

# ── KPIs ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

if not fc_7j.empty:
    col1.metric('Demain (GW)', f'{fc_7j["yhat"].iloc[0]:.1f}',
                f'IC [{fc_7j["yhat_lower"].iloc[0]:.1f} – {fc_7j["yhat_upper"].iloc[0]:.1f}]')
    max_7j = fc_7j['yhat'].max()
    col2.metric('Pic semaine (GW)', f'{max_7j:.1f}',
                f'le {fc_7j.loc[fc_7j["yhat"].idxmax(), "ds"].strftime("%a %d %b")}')

if not fc_30j.empty:
    col3.metric('Moy. 30 jours (GW)', f'{fc_30j["yhat"].mean():.1f}')

if not df_temp.empty and 'temp' in df_temp.columns:
    col4.metric('Temp. demain (°C)', f'{df_temp["temp"].iloc[0]:.1f}°')

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    '📅 Prévision 7 jours',
    '📆 Prévision 30 jours',
    '📊 Réalisé vs Prévisions',
])

with tab1:
    fig_7j = make_forecast_chart(
        df_hist.tail(21), fc_7j,
        title='Prévision J+7  Consommation France (GW)',
        color='rgb(46, 204, 113)'
    )
    st.plotly_chart(fig_7j, use_container_width=True)

    if not fc_7j.empty:
        st.dataframe(
            fc_7j[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                'ds': 'Date', 'yhat': 'Prévision (GW)',
                'yhat_lower': 'Borne basse', 'yhat_upper': 'Borne haute'
            }).style.format({'Prévision (GW)': '{:.2f}',
                             'Borne basse': '{:.2f}', 'Borne haute': '{:.2f}'}),
            use_container_width=True, hide_index=True
        )

with tab2:
    fig_30j = make_forecast_chart(
        df_hist, fc_30j,
        title='Prévision J+30  Consommation France (GW)',
        color='rgb(231, 76, 60)'
    )
    st.plotly_chart(fig_30j, use_container_width=True)

with tab3:
    st.subheader('Réalisé RTE vs Prévisions passées')

    col_a, col_b = st.columns([1, 3])
    with col_a:
        model_sel = st.radio('Modèle', ['7j', '30j'], horizontal=True)

    past_fcs  = load_past_forecasts(model_sel)
    all_runs  = sorted(past_fcs.keys(), reverse=True)

    if not all_runs:
        st.info("Aucune prévision datée disponible dans data/forecasts/")
    else:
        with col_a:
            selected_runs = st.multiselect(
                'Runs à afficher',
                options=all_runs,
                default=all_runs[:4],
            )

        df_rte_val = load_rte_actual(days=180)
        fig_val = make_validation_chart(df_rte_val, past_fcs, selected_runs, model_sel)
        st.plotly_chart(fig_val, use_container_width=True)

        # Tableau validation_log
        df_log = load_validation_log()
        if not df_log.empty:
            st.subheader('Historique des performances')
            df_log_model = df_log[df_log['model'] == model_sel].copy()
            if not df_log_model.empty:
                df_log_model = df_log_model.sort_values('validation_date', ascending=False)
                st.dataframe(
                    df_log_model[['validation_date', 'run_date', 'mape', 'mae', 'n_days', 'alert']]
                    .rename(columns={
                        'validation_date': 'Date validation',
                        'run_date'       : 'Run prévu le',
                        'mape'           : 'MAPE (%)',
                        'mae'            : 'MAE (MW)',
                        'n_days'         : 'Jours validés',
                        'alert'          : 'Alerte',
                    })
                    .style.format({'MAPE (%)': '{:.2f}', 'MAE (MW)': '{:,.0f}'})
                    .map(lambda v: 'color: red' if v is True else '', subset=['Alerte']),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info(f"Aucune validation enregistrée pour le modèle {model_sel}")
        else:
            st.info("validation_log.csv absent, il sera créé au prochain run_weekly.py")

st.divider()

# ── Température prévue ────────────────────────────────────────────────────────
if not df_temp.empty and 'temp' in df_temp.columns:
    st.subheader('🌡️ Températures prévues (4 points ruraux pondérés)')
    fig_temp = go.Figure()
    if 'temp_min' in df_temp.columns and 'temp_max' in df_temp.columns:
        fig_temp.add_trace(go.Scatter(
            x=pd.concat([df_temp['ds'], df_temp['ds'][::-1]]),
            y=pd.concat([df_temp['temp_max'], df_temp['temp_min'][::-1]]),
            fill='toself', fillcolor='rgba(255,165,0,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Min-Max', showlegend=True
        ))
    fig_temp.add_trace(go.Scatter(
        x=df_temp['ds'], y=df_temp['temp'],
        name='Temp. moyenne', line=dict(color='darkorange', width=2),
    ))
    fig_temp.add_hline(y=15, line_dash='dot', line_color='royalblue',
                        annotation_text='Seuil chauffage (15°C)')
    fig_temp.update_layout(height=250, yaxis_title='°C', margin=dict(t=20, b=30))
    st.plotly_chart(fig_temp, use_container_width=True)

st.divider()

# ── À propos ──────────────────────────────────────────────────────────────────
with st.expander('ℹ️ À propos du modèle'):
    st.markdown("""
    **Architecture** : Facebook Prophet (GAM) avec régresseurs exogènes

    **Variables exogènes** :
    - HDD/CDD calculés sur temp_min et temp_max (non-linéarité)
    - HDD_mean/CDD_mean sur temp_mean (signal complémentaire)
    - Lag saisonnier (hiver=4j, été=2j), inertie thermique bâtiments
    - % élèves en vacances scolaires (3 zones A/B/C pondérées)

    **Données météo** : 4 points ruraux pondérés (hors îlots de chaleur)
    - Alençon 35%, Bar-le-Duc 30%, Périgueux 20%, Montélimar 15%

    **Fenêtre d'entraînement** : 2023-01-01 → aujourd'hui − 8 semaines (holdout)

    **Performances** :
    - Modèle 7j : MAPE production **2.90%** | CV Optuna 2.17% (biaisé sélection)
    - Modèle 30j : MAPE production **2.53%** (test 5 mois 2026)

    **Sources** : RTE eco2mix • Open-Meteo Archive API
    """)
