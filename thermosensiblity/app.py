"""
Thermosensibilité de la consommation électrique française.

Lancer avec :  streamlit run app.py

Structure :
- app.py                  : point d'entrée + navigation (cette page)
- app_common.py           : loaders, sidebar partagée, entête période, pied sources
- page_chauffage.py       : page « Chauffage (froid) »
- page_climatisation.py   : page « Climatisation (chaud) »
- thermosensibilite.py    : logique de calcul (aucune dépendance Streamlit)
"""

import streamlit as st

import app_common as ac
from page_courbe import page_courbe
from page_chauffage import page_chauffage
from page_climatisation import page_climatisation

st.set_page_config(page_title="Thermosensibilité électrique", layout="wide")

# Sidebar partagée (chemins + cases d'exclusion des périodes de distorsion)
params = ac.sidebar_parametres()

# Chargement (mis en cache) — commun aux deux pages
try:
    conso_reg, temp_reg = ac.charger(params.conso_path, params.temp_path)
except FileNotFoundError:
    st.error(
        "Fichiers introuvables. Vérifie les chemins dans la barre latérale "
        "(les CSV doivent être dans le dossier de l'app, ou donne un chemin complet)."
    )
    st.stop()


def _page_courbe():
    ac.entete_periode(conso_reg, temp_reg, params)
    page_courbe(conso_reg, temp_reg, params)
    ac.pied_sources()


def _page_froid():
    ac.entete_periode(conso_reg, temp_reg, params)
    page_chauffage(conso_reg, temp_reg, params)
    ac.pied_sources()


def _page_chaud():
    ac.entete_periode(conso_reg, temp_reg, params)
    page_climatisation(conso_reg, temp_reg, params)
    ac.pied_sources()


pg = st.navigation([
    st.Page(_page_courbe, title="Consommation vs température", icon="📈", default=True),
    st.Page(_page_froid, title="Chauffage (froid)", icon="❄️"),
    st.Page(_page_chaud, title="Climatisation (chaud)", icon="☀️"),
])
pg.run()
