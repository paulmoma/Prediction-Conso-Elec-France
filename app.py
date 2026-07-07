"""
app.py
Dashboard Streamlit, prévision de consommation électrique France.

Lancement : streamlit run app.py
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path

DATA_DIR     = Path('data')
FORECAST_DIR = DATA_DIR / 'forecasts'


# Palette : baseline neutre (réalisé), accents tièdes pour les horizons de prévision.
# Choix : le réalisé reste discret, la prévision 30j tire vers le chaud (plus d'incertitude).
COLOR_HIST  = '#33414f'   # réalisé / historique — ardoise
COLOR_7J    = '#2c8c99'   # prévision 7 jours — sarcelle
COLOR_30J   = '#c06c75'   # prévision 30 jours — argile
COLOR_TEMP  = '#e0a23a'   # température — ambre
COLOR_SPLIT = '#9aa7b0'   # repère début de prévision / seuils — gris

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# Config commune des graphes : pas de barre d'outils, mise en page sobre,
# grille horizontale seule, virgule décimale (format FR).
PLOTLY_CONFIG = {'displayModeBar': False}
BASE_LAYOUT = dict(
    template='plotly_white',
    font=dict(family=FONT, size=12),
    separators=', ',                       # décimale = virgule, milliers = espace
    xaxis=dict(showgrid=False),
    hoverlabel=dict(font_family=FONT),
)

# Configuration page
st.set_page_config(
    page_title  = 'Prévision conso élec — France',
    page_icon   = '⚡',          # favicon de l'onglet uniquement
    layout      = 'wide',
)

# Quelques retouches sobres : largeur contenue, titre un peu plus dense.
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.6rem; max-width: 1200px;}
    h1 {font-weight: 650; letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _rgba(hex_color: str, alpha: float) -> str:
    """Convertit un hex (#rrggbb) en rgba(...) avec transparence."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# Loaders
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
    """Charge le réalisé RTE depuis rte_clean.csv (data lake mis à jour par run_weekly)."""
    clean_path = DATA_DIR / 'rte_clean.csv'
    if clean_path.exists():
        df = pd.read_csv(clean_path, parse_dates=['ds'])
        df = df.sort_values('ds').drop_duplicates('ds')
        df['y'] /= 1e3
        cutoff = pd.Timestamp(date.today() - timedelta(days=days))
        return df[df['ds'] >= cutoff].copy()

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


@st.cache_data(ttl=3600)
def load_temp_history() -> pd.DataFrame:
    path = DATA_DIR / 'temp_history.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['ds'])


@st.cache_data(ttl=3600)
def load_temp_forecast_log() -> pd.DataFrame:
    path = DATA_DIR / 'temp_forecast_log.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['ds', 'run_date'])


def compute_past_mapes(past_fcs: dict, df_rte: pd.DataFrame) -> pd.DataFrame:
    """Calcule MAPE et MAE pour chaque run passé sur les jours réalisés disponibles."""
    rows = []
    for run_date, df_fc in sorted(past_fcs.items()):
        df_val = df_fc.merge(df_rte[['ds', 'y']], on='ds', how='inner').dropna()
        if len(df_val) == 0:
            continue
        mape = (abs(df_val['y'] - df_val['yhat']) / abs(df_val['y'])).mean() * 100
        mae  = abs(df_val['y'] - df_val['yhat']).mean() * 1000  # GW → MW
        rows.append({
            'Run'          : run_date,
            'MAPE (%)'     : round(mape, 2),
            'MAE (MW)'     : round(mae),
            'Jours validés': len(df_val),
        })
    return pd.DataFrame(rows).sort_values('Run', ascending=False).reset_index(drop=True)


# Chart helpers
def make_forecast_chart(df_hist: pd.DataFrame,
                         df_fc: pd.DataFrame,
                         title: str,
                         color: str = COLOR_7J) -> go.Figure:
    fig = go.Figure()

    if not df_hist.empty:
        fig.add_trace(go.Scatter(
            x=df_hist['ds'], y=df_hist['y'],
            name='Historique', line=dict(color=COLOR_HIST, width=1.5),
            mode='lines'
        ))

    if df_fc.empty:
        st.warning("Aucune prévision disponible, lance d'abord run_weekly.py")
        return fig

    fig.add_trace(go.Scatter(
        x=pd.concat([df_fc['ds'], df_fc['ds'][::-1]]),
        y=pd.concat([df_fc['yhat_upper'], df_fc['yhat_lower'][::-1]]),
        fill='toself', fillcolor=_rgba(color, 0.15),
        line=dict(color='rgba(0,0,0,0)'),
        name='IC 80%', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=df_fc['ds'], y=df_fc['yhat'],
        name='Prévision', line=dict(color=color, width=2.5),
        mode='lines'
    ))
    split = df_fc['ds'].min()
    fig.add_vline(x=split, line_dash='dot', line_color=COLOR_SPLIT, opacity=0.7)

    # Annotation si gap entre fin du réalisé et début de prévision
    if not df_hist.empty:
        last_rte = df_hist['ds'].max()
        gap_days = (split - last_rte).days
        if gap_days > 1:
            mid_gap = last_rte + (split - last_rte) / 2
            fig.add_annotation(
                x=mid_gap, y=0, yref='paper', yanchor='top',
                text=f'données RTE non publiées ({gap_days}j)',
                showarrow=False, font=dict(size=10, color=COLOR_SPLIT),
            )

    fig.update_layout(
        **BASE_LAYOUT,
        title=title, yaxis_title='GW',
        hovermode='x unified', height=350,
        legend=dict(orientation='h', y=1.02),
        margin=dict(t=60, b=55),
    )
    return fig


def make_temp_figure(df_temp: pd.DataFrame, title: str = '') -> go.Figure:
    fig = go.Figure()
    if 'temp_min' in df_temp.columns and 'temp_max' in df_temp.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_temp['ds'], df_temp['ds'][::-1]]),
            y=pd.concat([df_temp['temp_max'], df_temp['temp_min'][::-1]]),
            fill='toself', fillcolor=_rgba(COLOR_TEMP, 0.15),
            line=dict(color='rgba(0,0,0,0)'), name='Min-Max', showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=df_temp['ds'], y=df_temp['temp'],
        name='Temp. moyenne', line=dict(color=COLOR_TEMP, width=2),
    ))
    fig.add_hline(y=15, line_dash='dot', line_color=COLOR_SPLIT,
                  annotation_text='Seuil chauffage (15°C)')
    fig.update_layout(
        **BASE_LAYOUT,
        title=title, height=220, yaxis_title='°C',
        margin=dict(t=30, b=30),
    )
    return fig


def make_past_forecast_chart(df_rte: pd.DataFrame,
                              df_fc: pd.DataFrame,
                              run_date: str,
                              model: str,
                              x_range: list) -> go.Figure:
    fig = go.Figure()
    label = '7 jours' if model == '7j' else '30 jours'
    color = COLOR_7J if model == '7j' else COLOR_30J

    fc_start = df_fc['ds'].min()

    df_win = df_rte[(df_rte['ds'] >= x_range[0]) & (df_rte['ds'] <= x_range[1])]
    if not df_win.empty:
        fig.add_trace(go.Scatter(
            x=df_win['ds'], y=df_win['y'],
            name='Réalisé', line=dict(color=COLOR_HIST, width=1.5), mode='lines'
        ))

    fig.add_trace(go.Scatter(
        x=pd.concat([df_fc['ds'], df_fc['ds'][::-1]]),
        y=pd.concat([df_fc['yhat_upper'], df_fc['yhat_lower'][::-1]]),
        fill='toself', fillcolor=_rgba(color, 0.15),
        line=dict(color='rgba(0,0,0,0)'), name='IC 80%', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=df_fc['ds'], y=df_fc['yhat'],
        name=f'Prévision du {run_date}',
        line=dict(color=color, width=2, dash='dash'), mode='lines'
    ))

    fig.add_vline(x=fc_start, line_dash='dot', line_color=COLOR_SPLIT, opacity=0.7)
    fig.add_annotation(
        x=fc_start, y=0, yref='paper', yanchor='top',
        text='date du run', showarrow=False,
        font=dict(size=10, color=COLOR_SPLIT),
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title=f'Prévision {label} du {run_date} — réalisé vs prévu',
        yaxis_title='GW', hovermode='x unified', height=360,
        legend=dict(orientation='h', y=1.02), margin=dict(l=60, t=60, b=45),
    )
    fig.update_xaxes(range=x_range)
    return fig


def make_past_temp_chart(df_actual_temp: pd.DataFrame,
                          df_temp_run: pd.DataFrame,
                          fc_start: pd.Timestamp,
                          x_range: list,
                          title: str = 'Température (moyenne) au run : prévu vs réalisé') -> go.Figure:
    fig = go.Figure()

    df_win = df_actual_temp[
        (df_actual_temp['ds'] >= x_range[0]) & (df_actual_temp['ds'] <= x_range[1])
    ]
    df_before = df_win[df_win['ds'] <= fc_start]
    df_during = df_win[df_win['ds'] >= fc_start]

    if not df_before.empty:
        fig.add_trace(go.Scatter(
            x=df_before['ds'], y=df_before['temp'],
            name='Observé (avant run)', line=dict(color=COLOR_HIST, width=1.5), mode='lines'
        ))
    if not df_during.empty:
        fig.add_trace(go.Scatter(
            x=df_during['ds'], y=df_during['temp'],
            name='Observé (après run)', line=dict(color=COLOR_HIST, width=2, dash='dot'), mode='lines'
        ))
    if not df_temp_run.empty:
        df_temp_run = df_temp_run[df_temp_run['ds'] >= fc_start]
        fig.add_trace(go.Scatter(
            x=df_temp_run['ds'], y=df_temp_run['temp'],
            name='Prévisions futures à la date du run',
            line=dict(color=COLOR_TEMP, width=2, dash='dash'), mode='lines'
        ))

    fig.add_vline(x=fc_start, line_dash='dot', line_color=COLOR_SPLIT, opacity=0.7)
    fig.add_annotation(
        x=fc_start, y=0, yref='paper', yanchor='top',
        text='date du run', showarrow=False,
        font=dict(size=10, color=COLOR_SPLIT),
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title=title, height=240, yaxis_title='°C', hovermode='x unified',
        margin=dict(l=60, t=40, b=45),
        legend=dict(orientation='h', y=1.02),
    )
    fig.update_xaxes(range=x_range)
    return fig


# Page principale

st.title('Prévision de consommation électrique journalière — France')
st.caption(
    f"Mise à jour du {date.today().strftime('%d %B %Y')} · "
    f"modèle Prophet avec degrés-jours de chauffe et vacances scolaires"
)

fc_7j    = load_forecast('7j')
fc_30j   = load_forecast('30j')
df_hist  = load_rte_actual(days=60)
df_temp  = load_temp_forecast()
df_vlog  = load_validation_log()

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)

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

if not df_vlog.empty:
    m7  = df_vlog[df_vlog['model'] == '7j' ]['mape'].tail(10).mean()
    m30 = df_vlog[df_vlog['model'] == '30j']['mape'].tail(10).mean()
    col5.metric('MAPE production', f'{m7:.1f} % / {m30:.1f} %',
                help='Moyenne glissante 10 derniers runs, modèles 7j / 30j')
else:
    col5.metric('MAPE production', '—', help='Données de validation non disponibles')

st.divider()

# Onglets
tab1, tab2, tab3 = st.tabs([
    'Prévision 7 jours',
    'Prévision 30 jours',
    'Réalisé vs prévisions',
])

with tab1:
    fig_7j = make_forecast_chart(
        df_hist.tail(21), fc_7j,
        title='Prévision J+7 — consommation France (GW)',
        color=COLOR_7J
    )
    st.plotly_chart(fig_7j, use_container_width=True, config=PLOTLY_CONFIG)

    if not df_temp.empty and 'temp' in df_temp.columns:
        st.subheader('Températures prévues')
        st.caption('4 points ruraux pondérés, hors îlots de chaleur urbains')
        st.plotly_chart(make_temp_figure(df_temp), use_container_width=True, config=PLOTLY_CONFIG, key='temp_7j')

with tab2:
    fig_30j = make_forecast_chart(
        df_hist, fc_30j,
        title='Prévision J+30 — consommation France (GW)',
        color=COLOR_30J
    )
    st.plotly_chart(fig_30j, use_container_width=True, config=PLOTLY_CONFIG)
    if not df_temp.empty and 'temp' in df_temp.columns:
        st.subheader('Températures prévues')
        st.caption('4 points ruraux pondérés, hors îlots de chaleur urbains')
        st.plotly_chart(make_temp_figure(df_temp), use_container_width=True, config=PLOTLY_CONFIG, key='temp_30j')

with tab3:
    c1, c2, _ = st.columns([1.1, 1.6, 2.3])
    with c1:
        model_sel = st.radio('Modèle', ['7j', '30j'], horizontal=True)

    past_fcs = load_past_forecasts(model_sel)
    all_runs = sorted(past_fcs.keys(), reverse=True)

    if not all_runs:
        st.info("Aucune prévision datée disponible dans data/forecasts/")
    else:
        with c2:
            # Défaut sur l'avant-dernier run : le dernier n'a pas encore de réalisé à comparer.
            cutoff = str(date.today() - timedelta(days=5))
            older  = [r for r in all_runs if r <= cutoff]
            default_run = older[0] if older else all_runs[-1]
            selected_run = st.selectbox('Run', options=all_runs,
                                        index=all_runs.index(default_run))

        df_rte_val = load_rte_actual(days=180)
        df_fc_sel  = past_fcs[selected_run]

        fc_start     = df_fc_sel['ds'].min()
        fc_end       = df_fc_sel['ds'].max()
        window_start = fc_start - timedelta(days=14)
        x_range      = [window_start, fc_end]

        fig_past = make_past_forecast_chart(df_rte_val, df_fc_sel, selected_run, model_sel, x_range)
        st.plotly_chart(fig_past, use_container_width=True, config=PLOTLY_CONFIG)

        df_temp_hist = load_temp_history()
        df_temp_log  = load_temp_forecast_log()
        run_ts       = pd.Timestamp(selected_run)
        df_temp_run  = df_temp_log[df_temp_log['run_date'].dt.normalize() == run_ts.normalize()] if not df_temp_log.empty else pd.DataFrame()
        if not df_temp_hist.empty or not df_temp_run.empty:
            fig_temp = make_past_temp_chart(df_temp_hist, df_temp_run, fc_start, x_range)
            st.plotly_chart(fig_temp, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader('Performances par run')
        df_perf = compute_past_mapes(past_fcs, df_rte_val)
        if not df_perf.empty:
            st.dataframe(
                df_perf.style
                    .format({'MAPE (%)': '{:.2f}', 'MAE (MW)': '{:,.0f}'})
                    .map(lambda v: 'color: #c06c75; font-weight: bold' if isinstance(v, float) and v > 5 else '', subset=['MAPE (%)']),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Pas encore de données réalisées disponibles pour calculer les performances.")



# À propos
with st.expander('À propos du modèle'):
    st.markdown("""
Modèle Facebook Prophet (GAM) avec variables externes.

Variables externes : indicateurs thermiques chaud/froid sur les températures min, max et
moyenne ; lag saisonnier (4 jours l'hiver, 2 l'été) pour l'inertie thermique des bâtiments ;
part d'élèves en vacances scolaires, zones A/B/C pondérées.

Données météo : 4 points ruraux pondérés, choisis hors îlots de chaleur urbains pour
représenter les principaux climats français : Alençon 35 %, Bar-le-Duc 30 %,
Périgueux 20 %, Montélimar 15 %.

Entraînement : du 1er janvier 2023 à aujourd'hui moins 8 semaines (test set).

Les performances en production sont consultables dans l'onglet *Réalisé vs prévisions*.

Sources : data.rte-france.fr · Open-Meteo Archive API.
    """)
