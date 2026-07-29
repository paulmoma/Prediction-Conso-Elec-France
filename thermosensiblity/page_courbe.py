"""Page « Consommation vs température » : la courbe en U au niveau national."""

import streamlit as st

import thermosensibilite as th
import app_common as ac


def page_courbe(conso_reg, temp_reg, params):
    st.title("Consommation électrique nationale vs température")
    st.caption(
        "Chaque point est une journée : sa température nationale (moyenne pondérée par la "
        "population) en abscisse, la consommation moyenne en France en ordonnée. La courbe rouge est la "
        "consommation moyenne par degré."
    )

    exc = dict(exclure_covid=params.exclure_covid, exclure_crise=params.exclure_crise)
    points = th.points_conso_temp(conso_reg, temp_reg, region=None, **exc)

    ac.courbe_u(points, seuil_froid=th.SEUIL_CHAUFFE, seuil_chaud=th.SEUIL_CHAUD)

