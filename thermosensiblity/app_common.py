"""
Éléments partagés par les pages de l'app Streamlit (loaders, sidebar, entête, pied).
La logique de calcul reste dans thermosensibilite.py ; ici on ne fait que de l'UI.
"""

import json
from types import SimpleNamespace

import altair as alt
import pandas as pd
import streamlit as st

import thermosensibilite as th
from thermosensibilite import normalize_region

alt.data_transformers.disable_max_rows()  # nos nuages font quelques milliers de points

CONSO_DEFAULT = "conso_journaliere_regionale_2013_2026.csv"
TEMP_DEFAULT = "temperature-quotidienne-regionale.csv"
GEOJSON_DEFAULT = "regions.geojson"
RTE_REFERENCE = -2400.0


@st.cache_data(show_spinner="Chargement des données…")
def charger(conso_path, temp_path):
    return th.charger_donnees(conso_path, temp_path)


@st.cache_data(show_spinner=False)
def charger_geojson(path):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        feat["properties"]["region_norm"] = normalize_region(feat["properties"]["nom"])
    return gj


def sidebar_parametres():
    """Barre latérale partagée : chemins de fichiers + cases d'exclusion des périodes.

    Les cases sont formulées en 'Inclure …' : cochées, la période est prise en compte ;
    décochées (défaut), elle est exclue de l'analyse.
    """
    with st.sidebar:
        st.header("Données")
        conso_path = st.text_input("Fichier consommation", CONSO_DEFAULT, key="conso_path")
        temp_path = st.text_input("Fichier température", TEMP_DEFAULT, key="temp_path")

        st.header("Périodes de distorsion")
        st.caption(
            "Ces périodes découplent la consommation de la météo (arrêts d'activité, "
            "sobriété). Décochées, elles sont exclues des régressions."
        )
        inc_covid = st.checkbox("Inclure le confinement Covid (mars–juin 2020)",
                                value=False, key="inc_covid")
        inc_crise = st.checkbox("Inclure la crise énergie 2022–2023 (guerre en Ukraine)",
                                value=False, key="inc_crise")

    return SimpleNamespace(
        conso_path=conso_path, temp_path=temp_path,
        exclure_covid=not inc_covid, exclure_crise=not inc_crise,
    )


def entete_periode(conso_reg, temp_reg, params):
    """Bandeau en haut de page : période réellement analysée + exclusions actives."""
    debut = max(conso_reg["Date"].min(), temp_reg["Date"].min())
    fin = min(conso_reg["Date"].max(), temp_reg["Date"].max())

    exclusions = []
    if params.exclure_covid:
        exclusions.append("Covid (mars–juin 2020)")
    if params.exclure_crise:
        exclusions.append("crise énergie 2022–2023")

    txt = f"**Période analysée :** {debut:%d/%m/%Y} → {fin:%d/%m/%Y}"
    if exclusions:
        txt += "  ·  exclusions actives : " + ", ".join(exclusions)
    else:
        txt += "  ·  aucune exclusion (toutes les données incluses)"
    st.caption(txt)


def pied_sources():
    """Pied de page : origine des données."""
    st.divider()
    st.caption(
        "**Sources.** Consommation électrique : Open Data Réseaux Énergies (ODRÉ / RTE), "
        "consommation quotidienne régionale. Température : ODRÉ, température quotidienne "
        "régionale (mesures du réseau Météo-France). Fond de carte : france-geojson "
        "(données IGN/INSEE). Référence de thermosensibilité −2 400 MW/°C : bilans "
        "électriques RTE."
    )

def courbe_u(points, titre=None, seuil_froid=None, seuil_chaud=None, segment=None):
    if points.empty:
        st.info("Pas de données à afficher pour cette sélection.")
        return

    moyenne = th.moyenne_par_degre(points)
    nuage = (
        alt.Chart(points).mark_circle(size=18, opacity=0.12, color="#5a7ea6").encode(
            x=alt.X("temp:Q", title="Température (°C)"),
            y=alt.Y("conso_mw:Q", title="Consommation (MW)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("temp:Q", title="°C", format=".1f"),
                     alt.Tooltip("conso_mw:Q", title="MW", format=",.0f")],
        )
    )
    ligne = alt.Chart(moyenne).mark_line(color="#c0392b", size=2.5).encode(x="temp:Q", y="conso_mw:Q")
    couches = [nuage, ligne]

    # segment de régression (le gradient mesuré), en vert, sur sa zone de validité
    if segment is not None and not segment.empty:
        couches.append(
            alt.Chart(segment).mark_line(color="#27ae60", size=3, strokeDash=[2, 0])
            .encode(x="temp:Q", y="conso_mw:Q")
        )

    for seuil, couleur in [(seuil_froid, "#2c6fbb"), (seuil_chaud, "#d35400")]:
        if seuil is not None:
            couches.append(
                alt.Chart(pd.DataFrame({"x": [seuil]}))
                .mark_rule(color=couleur, strokeDash=[5, 4]).encode(x="x:Q")
            )

    chart = alt.layer(*couches)
    if titre:
        chart = chart.properties(title=titre)
    st.altair_chart(chart, use_container_width=True)
    

def segment_gradient(points, pente, seuil, sens):
    """Segment de droite matérialisant le gradient, tracé sur sa zone de validité."""
    if pente is None or points.empty:
        return None
    branche = points[points["temp"] < seuil] if sens == "froid" else points[points["temp"] > seuil]
    branche = branche.dropna(subset=["temp", "conso_mw"])
    if len(branche) < 5:
        return None
    x0, x1 = branche["temp"].min(), branche["temp"].max()
    # ordonnée à l'origine telle que la droite passe par le centroïde (comme une OLS)
    a = branche["conso_mw"].mean() - pente * branche["temp"].mean()
    return pd.DataFrame({"temp": [x0, x1], "conso_mw": [a + pente * x0, a + pente * x1]})