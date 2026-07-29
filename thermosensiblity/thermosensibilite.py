"""
Calcul de la thermosensibilité de la consommation électrique régionale française.

Module de logique pure (pas de Streamlit ici) : chargement, agrégation,
pondération et régressions. Importable aussi bien par l'app Streamlit que
par un notebook, pour garder une seule source de vérité sur les calculs.

Fichiers d'entrée attendus (mêmes conventions que le notebook) :
- conso : 'conso_journaliere_regionale_2013_2026.csv', séparateur ',',
          colonne d'index en position 0, colonnes ['Date', 'Région', 'conso_mw'].
- temp  : 'temperature-quotidienne-regionale.csv' (ODRÉ), séparateur ';',
          format long, colonnes dont ['Date', 'Région', 'TMoy (°C)'].
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- Paramètres par défaut -------------------------------------------------

SEUIL_CHAUFFE = 15.0     # °C : seuil RTE sous lequel le chauffage électrique domine
MIN_OBS = 20             # nb minimum d'observations pour qu'une régression soit renvoyée

# Périodes de distorsion structurelle (conso découplée de la météo), exclues par défaut
PERIODE_COVID = ("2020-03-17", "2020-06-02")   # confinement Covid 1
PERIODE_CRISE = ("2022-09-01", "2023-06-30")   # crise énergie / sobriété (guerre en Ukraine)

# Poids population approximatifs par région (ordres de grandeur INSEE 2023).
# À remplacer par les chiffres exacts du dernier millésime pour un rapport rigoureux.
POIDS_REGION = {
    "Île-de-France":               0.191,
    "Auvergne-Rhône-Alpes":        0.126,
    "Hauts-de-France":             0.097,
    "Provence-Alpes-Côte d'Azur":  0.102,
    "Nouvelle-Aquitaine":          0.094,
    "Occitanie":                   0.093,
    "Grand Est":                   0.085,
    "Pays de la Loire":            0.060,
    "Normandie":                   0.055,
    "Bretagne":                    0.052,
    "Bourgogne-Franche-Comté":     0.042,
    "Centre-Val de Loire":         0.040,
    "Corse":                       0.005,   # présent en température, absent de la conso
}


# --- Utilitaires -----------------------------------------------------------

def normalize_region(name: str) -> str:
    """Normalise un nom de région (accents, tirets, casse) pour fiabiliser les jointures."""
    if not isinstance(name, str):
        return name
    nfkd = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    s = re.sub(r"[-']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_POIDS_NORM = {normalize_region(k): v for k, v in POIDS_REGION.items()}


def filtrer_periodes(df: pd.DataFrame, exclure_covid: bool = True,
                     exclure_crise: bool = True, col_date: str = "Date") -> pd.DataFrame:
    """Retire, au choix, la période Covid et/ou la période de crise énergétique."""
    masque = pd.Series(False, index=df.index)
    if exclure_covid:
        d, f = PERIODE_COVID
        masque |= (df[col_date] >= d) & (df[col_date] <= f)
    if exclure_crise:
        d, f = PERIODE_CRISE
        masque |= (df[col_date] >= d) & (df[col_date] <= f)
    return df[~masque].copy()


# --- Chargement ------------------------------------------------------------

def charger_donnees(conso_path: str, temp_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge conso + température et renvoie deux dataframes régionaux nettoyés.

    Returns
    -------
    conso_reg : ['Date', 'Région', 'region_norm', 'conso_mw']
    temp_reg  : ['Date', 'Région', 'region_norm', 'temp', 'temp_min', 'temp_max', 'poids']
                ('temp' = température moyenne ; temp_min/temp_max présents si le fichier
                 fournit TMin/TMax — utiles notamment pour le signal clim, piloté par le pic
                 de chaleur du jour plutôt que par la moyenne).
    """
    conso_reg = pd.read_csv(conso_path, parse_dates=["Date"], index_col=0)
    conso_reg = conso_reg[["Date", "Région", "conso_mw"]].copy()
    conso_reg["region_norm"] = conso_reg["Région"].apply(normalize_region)

    temp_reg = pd.read_csv(temp_path, sep=";", parse_dates=["Date"])
    temp_reg = temp_reg.rename(columns={
        "TMoy (°C)": "temp", "TMin (°C)": "temp_min", "TMax (°C)": "temp_max"})
    temp_reg["region_norm"] = temp_reg["Région"].apply(normalize_region)
    temp_reg["poids"] = temp_reg["region_norm"].map(_POIDS_NORM)
    temp_reg = temp_reg.dropna(subset=["poids", "temp"])

    cols = ["Date", "Région", "region_norm", "temp", "poids"]
    for extra in ("temp_min", "temp_max"):   # gardés seulement s'ils existent dans le fichier
        if extra in temp_reg.columns:
            cols.insert(cols.index("poids"), extra)
    temp_reg = temp_reg[cols]

    return conso_reg, temp_reg


def construire_national(conso_reg: pd.DataFrame, temp_reg: pd.DataFrame,
                        ponderation: str = "population",
                        exclure_covid: bool = True,
                        exclure_crise: bool = True) -> pd.DataFrame:
    """Agrège au niveau France : conso = somme des régions, température = moyenne pondérée.

    ponderation : 'population' (poids INSEE) ou 'simple' (moyenne non pondérée).
    exclure_covid / exclure_crise : retire ou non ces périodes de distorsion.
    """
    conso_nat = conso_reg.groupby("Date", as_index=False)["conso_mw"].sum()

    if ponderation == "simple":
        temp_nat = temp_reg.groupby("Date", as_index=False)["temp"].mean()
    else:
        temp_nat = (
            temp_reg.groupby("Date")
            .apply(lambda g: np.average(g["temp"], weights=g["poids"]), include_groups=False)
            .reset_index(name="temp")
        )

    national = conso_nat.merge(temp_nat, on="Date", how="inner")
    national["weekday"] = national["Date"].dt.weekday
    national = filtrer_periodes(national, exclure_covid, exclure_crise)
    return national.sort_values("Date").reset_index(drop=True)


# --- Régressions -----------------------------------------------------------

def gradient(national: pd.DataFrame, seuil: float = SEUIL_CHAUFFE) -> dict:
    """Régression conso ~ temp + jour de semaine, sous le seuil de chauffe.

    Returns dict(gradient_mw_par_c, r2, n).
    """
    g = national[national["temp"] < seuil]
    if len(g) < MIN_OBS:
        return {"gradient_mw_par_c": np.nan, "r2": np.nan, "n": len(g)}
    X = pd.concat(
        [g[["temp"]], pd.get_dummies(g["weekday"], prefix="wd", drop_first=True)],
        axis=1,
    ).astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(g["conso_mw"], X, missing="drop").fit()
    return {"gradient_mw_par_c": float(model.params["temp"]),
            "r2": float(model.rsquared), "n": int(len(g))}


def balayage_seuil(national: pd.DataFrame,
                   seuils=(12, 13, 14, 15, 16, 17, 18, 20)) -> pd.DataFrame:
    """Gradient national pour une série de seuils de température."""
    lignes = []
    for s in seuils:
        r = gradient(national, seuil=s)
        lignes.append({"seuil_C": s, **r})
    return pd.DataFrame(lignes)


def gradient_par_region(conso_reg: pd.DataFrame, temp_reg: pd.DataFrame,
                        seuil: float = SEUIL_CHAUFFE,
                        exclure_covid: bool = True,
                        exclure_crise: bool = True) -> pd.DataFrame:
    """Gradient région par région + indicateurs de fiabilité (jours sous seuil).

    Colonnes : Région, gradient_mw_par_c, gradient_pct_par_c, conso_moy_hiver_mw,
               r2, jours_sous_seuil, jours_total, pct_jours_froids.
    """
    reg = pd.merge(
        conso_reg[["Date", "Région", "region_norm", "conso_mw"]],
        temp_reg[["Date", "region_norm", "temp"]],
        on=["Date", "region_norm"], how="inner",
    )
    reg["weekday"] = reg["Date"].dt.weekday
    reg = filtrer_periodes(reg, exclure_covid, exclure_crise)

    lignes = []
    for region, g in reg.groupby("Région", observed=True):
        sub = g[g["temp"] < seuil]
        n_total, n_froid = len(g), len(sub)
        pct = 100 * n_froid / n_total if n_total else np.nan
        base = {"Région": region, "jours_sous_seuil": n_froid,
                "jours_total": n_total, "pct_jours_froids": pct}
        if n_froid < MIN_OBS:
            lignes.append({**base, "gradient_mw_par_c": np.nan,
                           "gradient_pct_par_c": np.nan,
                           "conso_moy_hiver_mw": np.nan, "r2": np.nan})
            continue
        X = pd.concat(
            [sub[["temp"]], pd.get_dummies(sub["weekday"], prefix="wd", drop_first=True)],
            axis=1,
        ).astype(float)
        X = sm.add_constant(X)
        model = sm.OLS(sub["conso_mw"], X, missing="drop").fit()
        pente = float(model.params["temp"])
        conso_moy = float(sub["conso_mw"].mean())
        lignes.append({
            **base,
            "gradient_mw_par_c": pente,
            "gradient_pct_par_c": 100 * pente / conso_moy if conso_moy else np.nan,
            "conso_moy_hiver_mw": conso_moy,
            "r2": float(model.rsquared),
        })

    return pd.DataFrame(lignes).sort_values("gradient_pct_par_c").reset_index(drop=True)


def evolution_saisonniere(national: pd.DataFrame,
                          seuil: float = SEUIL_CHAUFFE) -> pd.DataFrame:
    """Gradient national hiver par hiver (déc + janv-fév suivants)."""
    df = national.copy()
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["saison_hiver"] = np.where(
        df["month"] == 12, df["year"],
        np.where(df["month"].isin([1, 2]), df["year"] - 1, np.nan),
    )
    lignes = []
    for saison, g in df.dropna(subset=["saison_hiver"]).groupby("saison_hiver"):
        r = gradient(g, seuil=seuil)
        lignes.append({"saison_hiver": f"{int(saison)}-{int(saison) + 1}", **r})
    return pd.DataFrame(lignes)


# --- Thermosensibilité à la CHALEUR (climatisation) ------------------------

SEUIL_CHAUD = 22.0        # °C : seuil au-dessus duquel on estime le signal clim
MIN_JOURS_CHAUD = 30      # nb minimum de jours chauds pour une estimation fiable


def _merge_regional(conso_reg: pd.DataFrame, temp_reg: pd.DataFrame,
                    exclure_covid: bool = True, exclure_crise: bool = True) -> pd.DataFrame:
    """Table régionale conso + température (mean), jour de semaine ajouté."""
    reg = pd.merge(
        conso_reg[["Date", "Région", "region_norm", "conso_mw"]],
        temp_reg[["Date", "region_norm", "temp"]],
        on=["Date", "region_norm"], how="inner",
    )
    reg["weekday"] = reg["Date"].dt.weekday
    return filtrer_periodes(reg, exclure_covid, exclure_crise)


def gradient_chaud_par_region(conso_reg: pd.DataFrame, temp_reg: pd.DataFrame,
                              seuil: float = SEUIL_CHAUD,
                              min_jours: int = MIN_JOURS_CHAUD,
                              exclure_covid: bool = True,
                              exclure_crise: bool = True) -> pd.DataFrame:
    """Gradient clim (pente POSITIVE) par région, avec R² et fiabilité.

    Colonnes : Région, region_norm, gradient_mw_par_c, gradient_pct_par_c,
               r2, jours_chauds, fiable.
    """
    reg = _merge_regional(conso_reg, temp_reg, exclure_covid, exclure_crise)
    lignes = []
    for region, g in reg.groupby("Région", observed=True):
        sub = g[g["temp"] > seuil]
        n = len(sub)
        rn = g["region_norm"].iloc[0]
        if n < min_jours:
            lignes.append({"Région": region, "region_norm": rn,
                           "gradient_mw_par_c": np.nan, "gradient_pct_par_c": np.nan,
                           "r2": np.nan, "jours_chauds": n, "fiable": False})
            continue
        X = pd.concat(
            [sub[["temp"]], pd.get_dummies(sub["weekday"], prefix="wd", drop_first=True)],
            axis=1,
        ).astype(float)
        X = sm.add_constant(X)
        m = sm.OLS(sub["conso_mw"], X).fit()
        pente = float(m.params["temp"])
        conso_moy = float(sub["conso_mw"].mean())
        lignes.append({
            "Région": region, "region_norm": rn,
            "gradient_mw_par_c": pente,
            "gradient_pct_par_c": 100 * pente / conso_moy if conso_moy else np.nan,
            "r2": float(m.rsquared), "jours_chauds": n, "fiable": True,
        })
    return pd.DataFrame(lignes).sort_values("gradient_mw_par_c", ascending=False).reset_index(drop=True)


def gradient_chaud_annuel(conso_reg: pd.DataFrame, temp_reg: pd.DataFrame,
                          seuil: float = SEUIL_CHAUD,
                          min_jours: int = MIN_JOURS_CHAUD,
                          exclure_covid: bool = True,
                          exclure_crise: bool = True) -> pd.DataFrame:
    """Gradient clim par région ET par année (format long).

    Colonnes : Région, annee, gradient_mw_par_c, n_jours.
    Renvoie NaN pour les cellules (région × année) sous min_jours jours chauds.
    """
    reg = _merge_regional(conso_reg, temp_reg, exclure_covid, exclure_crise)
    reg["annee"] = reg["Date"].dt.year
    lignes = []
    for (region, annee), g in reg.groupby(["Région", "annee"], observed=True):
        sub = g[g["temp"] > seuil]
        n = len(sub)
        if n < min_jours:
            lignes.append({"Région": region, "annee": int(annee),
                           "gradient_mw_par_c": np.nan, "n_jours": n})
            continue
        X = pd.concat(
            [sub[["temp"]], pd.get_dummies(sub["weekday"], prefix="wd", drop_first=True)],
            axis=1,
        ).astype(float)
        X = sm.add_constant(X)
        m = sm.OLS(sub["conso_mw"], X).fit()
        lignes.append({"Région": region, "annee": int(annee),
                       "gradient_mw_par_c": float(m.params["temp"]), "n_jours": n})
    return pd.DataFrame(lignes)


# --- Courbe consommation vs température (forme en U) ------------------------

def points_conso_temp(conso_reg: pd.DataFrame, temp_reg: pd.DataFrame,
                      region: str | None = None, ponderation: str = "population",
                      exclure_covid: bool = True, exclure_crise: bool = True) -> pd.DataFrame:
    """Points (Date, temp, conso_mw) pour tracer la courbe conso vs température.

    region=None -> niveau national (conso sommée, température pondérée population).
    region="…"  -> la région donnée (sa conso, sa température moyenne).
    """
    if region is None:
        nat = construire_national(conso_reg, temp_reg, ponderation,
                                  exclure_covid, exclure_crise)
        return nat[["Date", "temp", "conso_mw"]].copy()
    reg = _merge_regional(conso_reg, temp_reg, exclure_covid, exclure_crise)
    return reg.loc[reg["Région"] == region, ["Date", "temp", "conso_mw"]].copy()


def moyenne_par_degre(points: pd.DataFrame, largeur: float = 1.0) -> pd.DataFrame:
    """Consommation moyenne par tranche de température (pour la ligne lissée de la courbe)."""
    p = points.dropna(subset=["temp", "conso_mw"]).copy()
    p["temp"] = (p["temp"] / largeur).round() * largeur
    return (p.groupby("temp", as_index=False)["conso_mw"].mean()
            .sort_values("temp").reset_index(drop=True))
