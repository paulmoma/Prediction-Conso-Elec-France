"""Page « Chauffage (froid) » : thermosensibilité de la consommation au froid."""

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

import thermosensibilite as th
from thermosensibilite import normalize_region
import app_common as ac


def page_chauffage(conso_reg, temp_reg, params):
    st.title("Impact du chauffage électrique sur la consommation électrique")
    st.caption(
        "Gradient de consommation par degré perdu (MW/°C). Référence RTE : environ "
        "**−2 400 MW/°C** en hiver, sous 15 °C."
    )

    exc = dict(exclure_covid=params.exclure_covid, exclure_crise=params.exclure_crise)

    national = th.construire_national(conso_reg, temp_reg, ponderation="population", **exc)

    seuil = st.select_slider(
        "Seuil de température (°C)",
        options=[12, 13, 14, 15, 16, 17, 18, 20], value=15,
        help="Seuil sous lequel on estime le chauffage actif. RTE utilise 15 °C.",
    )

    # --- Gradient national ---
    res = th.gradient(national, seuil=seuil)
    grad_nat = res["gradient_mw_par_c"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gradient à {seuil} °C", f"{grad_nat:,.0f} MW/°C",
              f"{grad_nat - ac.RTE_REFERENCE:+,.0f} vs RTE", delta_color="off")
    c2.metric("Ajustement (R²)", f"{res['r2']:.3f}")
    c3.metric("Jours utilisés", f"{res['n']:,}")

    # --- Balayage du seuil ---
    st.subheader("Sensibilité au seuil")
    sweep = th.balayage_seuil(national).dropna(subset=["gradient_mw_par_c"])
    barres = (
        alt.Chart(sweep).mark_bar().encode(
            x=alt.X("seuil_C:O", title="Seuil (°C)"),
            y=alt.Y("gradient_mw_par_c:Q", title="Gradient (MW/°C)",
                    scale=alt.Scale(reverse=True)),
            color=alt.condition(alt.datum.seuil_C == seuil,
                                alt.value("#c0392b"), alt.value("#5a7ea6")),
            tooltip=[alt.Tooltip("seuil_C:O", title="Seuil °C"),
                     alt.Tooltip("gradient_mw_par_c:Q", title="MW/°C", format=",.0f"),
                     alt.Tooltip("r2:Q", title="R²", format=".3f")],
        )
    )
    ligne_rte = (alt.Chart(pd.DataFrame({"y": [ac.RTE_REFERENCE]}))
                 .mark_rule(strokeDash=[6, 4], color="#888").encode(y="y:Q"))
    st.altair_chart(barres + ligne_rte, use_container_width=True)

    # --- Régional ---
    st.subheader("Différences entre régions")
    metrique = st.radio(
        "Indicateur",
        ["gradient_pct_par_c", "gradient_mw_par_c"],
        format_func=lambda k: {"gradient_pct_par_c": "Relatif (%/°C)",
                               "gradient_mw_par_c": "Absolu (MW/°C)"}[k],
        horizontal=True,
    )
    regions = th.gradient_par_region(conso_reg, temp_reg, seuil=seuil, **exc)
    regions["region_norm"] = regions["Région"].apply(normalize_region)

    titre = "%/°C" if metrique == "gradient_pct_par_c" else "MW/°C"
    try:
        geojson = ac.charger_geojson(ac.GEOJSON_DEFAULT)
        fig = px.choropleth(
            regions, geojson=geojson, locations="region_norm",
            featureidkey="properties.region_norm", color=metrique,
            color_continuous_scale="Reds_r", hover_name="Région",
            hover_data={"region_norm": False, metrique: ":.2f",
                        "pct_jours_froids": ":.0f", "r2": ":.2f"},
            labels={metrique: titre},
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar_title=titre)
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning(f"Carte non affichée : '{ac.GEOJSON_DEFAULT}' introuvable.")

    titre_axe = "Gradient (%/°C)" if metrique == "gradient_pct_par_c" else "Gradient (MW/°C)"
    chart_reg = (
        alt.Chart(regions).mark_bar(color="#5a7ea6").encode(
            x=alt.X(f"{metrique}:Q", title=titre_axe, scale=alt.Scale(reverse=True)),
            y=alt.Y("Région:N", sort="-x", title=None),
            tooltip=["Région",
                     alt.Tooltip("gradient_mw_par_c:Q", title="MW/°C", format=",.0f"),
                     alt.Tooltip("gradient_pct_par_c:Q", title="%/°C", format=".2f"),
                     alt.Tooltip("pct_jours_froids:Q", title="% jours froids", format=".0f"),
                     alt.Tooltip("r2:Q", title="R²", format=".2f")],
        )
    )
    st.altair_chart(chart_reg, use_container_width=True)

    with st.expander("Fiabilité : jours sous le seuil par région"):
        st.dataframe(
            regions[["Région", "jours_sous_seuil", "jours_total",
                     "pct_jours_froids", "r2"]]
            .sort_values("pct_jours_froids")
            .style.format({"pct_jours_froids": "{:.0f} %", "r2": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

    # --- Courbe conso vs température pour une région choisie ---
    st.subheader("Courbe consommation vs température (par région)")
    region_courbe = st.selectbox("Région à visualiser", sorted(regions["Région"].tolist()), key="region_courbe_froid")
    points = th.points_conso_temp(conso_reg, temp_reg, region=region_courbe, **exc)

    # segment matérialisant le gradient de la région, sur la branche froide
    pente = regions.loc[regions["Région"] == region_courbe, "gradient_mw_par_c"].iloc[0]
    seg = ac.segment_gradient(points, pente, seuil, "froid")

    ac.courbe_u(points, titre=region_courbe, seuil_froid=seuil, segment=seg)
    st.caption(
        "La branche descendante à gauche du repère bleu est le signal chauffage de la région. "
        "Sa pente correspond à la thermosensibilité au froid mesurée plus haut."
    )
