"""Page « Climatisation (chaud) » : thermosensibilité de la consommation à la chaleur."""

import altair as alt
import plotly.express as px
import streamlit as st

import thermosensibilite as th
import app_common as ac


def page_climatisation(conso_reg, temp_reg, params):
    st.title("Climatisation — thermosensibilité à la chaleur")
    st.caption(
        "Quand il fait chaud, la consommation remonte (clim, ventilation). Le signal est "
        "faible en France et n'est fiable que là où s'accumulent assez de jours chauds. On "
        "n'affiche sur la carte que les régions dont le R² dépasse le seuil choisi."
    )

    exc = dict(exclure_covid=params.exclure_covid, exclure_crise=params.exclure_crise)

    r2_min = st.slider(
        "R² minimum pour afficher une région", 0.0, 0.9, 0.6, 0.05,
        help="En dessous de ce seuil de qualité d'ajustement, la région est masquée.",
    )

    chaud = th.gradient_chaud_par_region(conso_reg, temp_reg, seuil=th.SEUIL_CHAUD, **exc)
    chaud_fiable = chaud[(chaud["fiable"]) & (chaud["r2"] >= r2_min)].copy()

    if chaud_fiable.empty:
        st.info("Aucune région n'atteint ce seuil de R². Baisse le curseur pour en afficher.")
        return

    # --- Carte : régions fiables colorées par gradient clim (positif) ---
    try:
        geojson = ac.charger_geojson(ac.GEOJSON_DEFAULT)
        fig = px.choropleth(
            chaud_fiable, geojson=geojson, locations="region_norm",
            featureidkey="properties.region_norm", color="gradient_mw_par_c",
            color_continuous_scale="OrRd", hover_name="Région",
            hover_data={"region_norm": False, "gradient_mw_par_c": ":.0f",
                        "r2": ":.2f", "jours_chauds": True},
            labels={"gradient_mw_par_c": "MW/°C"},
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar_title="MW/°C")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Régions avec R² ≥ {r2_min:.2f} (seuil chaud {th.SEUIL_CHAUD:.0f} °C). "
            "Plus c'est foncé, plus la consommation grimpe avec la chaleur. Les régions "
            "non fiables (nord/ouest, trop peu de jours chauds) sont volontairement absentes."
        )
    except FileNotFoundError:
        st.warning(f"Carte non affichée : '{ac.GEOJSON_DEFAULT}' introuvable.")

    # --- Évolution annuelle pour une région fiable ---
    st.subheader("Évolution année par année")
    region_choisie = st.selectbox(
        "Région à examiner (parmi les régions fiables)",
        chaud_fiable["Région"].tolist(),
    )

    annuel = th.gradient_chaud_annuel(conso_reg, temp_reg, seuil=th.SEUIL_CHAUD, **exc)
    serie = (annuel[annuel["Région"] == region_choisie]
             .dropna(subset=["gradient_mw_par_c"]).sort_values("annee"))

    if len(serie) < 2:
        st.info(
            f"{region_choisie} n'a pas assez d'années exploitables (jours chauds "
            "insuffisants) pour tracer une évolution."
        )
        return

    courbe = (
        alt.Chart(serie).mark_line(point=True, color="#d35400").encode(
            x=alt.X("annee:O", title="Année"),
            y=alt.Y("gradient_mw_par_c:Q", title="Gradient clim (MW/°C)"),
            tooltip=[alt.Tooltip("annee:O", title="Année"),
                     alt.Tooltip("gradient_mw_par_c:Q", title="MW/°C", format=".0f"),
                     alt.Tooltip("n_jours:Q", title="jours chauds")],
        )
    )
    st.altair_chart(courbe, use_container_width=True)
    st.caption(
        "⚠️ Une hausse ne prouve pas à elle seule l'adoption croissante de la "
        "climatisation : un été très chaud produit plus de jours chauds ET un gradient "
        "plus raide (la réponse clim s'accélère aux fortes chaleurs). À lire comme "
        "exploratoire."
    )

    # --- Courbe conso vs température de la région sélectionnée ---
    st.subheader(f"Courbe consommation vs température — {region_choisie}")
    points = th.points_conso_temp(conso_reg, temp_reg, region=region_choisie, **exc)
    ac.courbe_u(points, titre=region_choisie, seuil_chaud=th.SEUIL_CHAUD)
    st.caption(
        "La branche remontante à droite du repère orange est le signal climatisation de la "
        "région : c'est sa pente que mesure le gradient chaud."
    )
