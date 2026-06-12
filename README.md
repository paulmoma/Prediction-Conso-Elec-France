# Prévision consommation électrique France, modèle Prophet

Modèle de prévision de la consommation électrique nationale française
à horizons **7 jours** et **30 jours**, construit uniquement avec des données
publiques gratuites.

> Projet réalisé dans le cadre d'une formation Data Engineer (Campus Numérique in the Alps).  
> Voir aussi : [NEBCO - mécanisme d'effacement](https://github.com/paulmoma/Nebco---mecanisme-d-effacement)
> *(ce modèle fournit la baseline de consommation nécessaire au dispatch NEBCO)*

---

## Résultats

| Modèle | MAPE CV (Optuna) | MAPE production |
|--------|-----------------|-----------------|
| **Prophet 7j** | 2.17%* | **2.90%** |
| **Prophet 30j** | 2.37%* | **2.53%** |
| Naïf saisonnier | - | 8.9% |
| TimesFM zero-shot (200M) | - | 8.3% |

*MAPE CV optimiste : retient le meilleur résultat parmi 100 combinaisons d'hyperparamètres testées par Optuna (recherche bayésienne, plus efficace qu'un grid search aléatoire)*

**MAPE production** = rolling validation sur données jamais vues (2026).

---

## Méthodologie

### Données
- **Consommation** : [RTE eco2mix](https://www.rte-france.com/eco2mix) - granularité 15 min → agrégée en journalier
- **Météo** : [Open-Meteo](https://open-meteo.com) Archive API - temp mean/min/max pour les prévisions jusqu'à J+16, moyennes des données des 3 dernières années ensuite.

### Représentation spatiale de la température
4 points ruraux pondérés (hors îlots de chaleur urbains +1.47°C) :

| Point | Poids | Rôle |
|-------|-------|------|
| Alençon (NW) | 35% | Climat océanique, fort chauffage élec. |
| Bar-le-Duc (NE) | 30% | Climat continental, hivers froids |
| Périgueux (SW) | 20% | Climat tempéré |
| Montélimar (SE) | 15% | Pré-méditerranéen |

### Feature engineering
```
HDD_min  = max(0, seuil - temp_min)   → nuits froides, chauffage nocturne
CDD_max  = max(0, temp_max - seuil)   → pics chaleur, climatisation intensive
HDD_mean = max(0, seuil - temp_mean)  → journées froides, chauffage diurne
CDD_mean = max(0, temp_mean - seuil)  → journées tièdes, climatisation modérée
lag_hiver = temp_mean(t-4j) × hiver   → inertie thermique bâtiments (4 jours)
lag_ete   = temp_mean(t-2j) × été     → climatisation réactive (2 jours)
pct_vac   = Σ w_zone × vac_zone       → % élèves en vacances (7j uniquement)
```

### Optimisation
- Hyperparamètres optimisés par **Optuna TPE** (optimisation bayésienne, 100 trials)
- Cross-validation Prophet : `initial='730 days'`, `period='30 days'`
- Monitoring overfitting : **Mann-Whitney U** + écart relatif > 40%

---

## Installation

```bash
git clone https://github.com/paulmoma/energy-forecasting-france
cd energy-forecasting-france

conda create -n prophet python=3.11
conda activate prophet
pip install -r requirements.txt
```

### Données (non incluses dans le repo)

**Consommation RTE** : télécharger les fichiers `conso_mix_RTE_YYYY.xls`
sur [eco2mix RTE](https://www.rte-france.com/eco2mix/telecharger-les-indicateurs)
et les placer dans `data/`.

**Températures** : téléchargées automatiquement via Open-Meteo au premier run.

---

## Usage

```bash
# Réentraînement manuel
python retrain.py

# Run hebdomadaire (validation + prévisions + retrain)
python run_weekly.py

# Dashboard Streamlit
streamlit run app.py
```

### Automatisation (cron lundi + jeudi à 10h)
```bash
bash cron_setup.sh
```

---

## Structure du projet

```
├── src/
│   ├── data.py        # Chargement RTE + Open-Meteo, validation températures
│   ├── features.py    # HDD/CDD/lags/pct_vacances, BEST_PARAMS_7J et 30J
│   ├── model.py       # Build/train/predict Prophet 7j et 30j
│   └── evaluate.py    # Métriques, plots, test overfitting Mann-Whitney
├── retrain.py         # Réentraînement + évaluation holdout + promotion MLflow
├── run_weekly.py      # Pipeline hebdomadaire (validation + prévisions + retrain)
├── run_if_needed.sh   # Guard idempotence pour le cron
├── cron_setup.sh      # Configuration automatique du cron
├── app.py             # Dashboard Streamlit
├── requirements.txt
└── data/              # Non versionné (.gitignore)
    ├── conso_mix_RTE_YYYY.xls   # À télécharger depuis RTE
    ├── temp_Alencon.csv         # Cache Open-Meteo (auto-généré)
    └── forecast_7j_latest.csv   # Dernières prévisions (auto-généré)
```

---

## Note méthodologique

Le MAPE CV Optuna (2.17% / 2.37%) est biaisé par la **sélection sur 100 trials** :
Optuna évalue 100 combinaisons sur les mêmes folds et retient le minimum - ce minimum
est systématiquement trop optimiste. Les paramètres restent valides ; seule l'estimation
de performance l'est pas. Le MAPE de production honnête (2.90% / 2.53%) est évalué
sur un rolling CV indépendant ou sur des données jamais vues.

---

## Pistes abandonnées

| Piste | Résultat | Raison |
|-------|---------|--------|
| 8 villes pondérées | +2 pts MAPE | Biais îlots de chaleur urbains |
| Fenêtre 2007-2025 | 5.93% | Concept drift (COVID, crise énergie 2022) |
| Poids asymétriques hiver/été | 2.47% | Pas mieux que symétrique |
| Vacances scolaires (30j) | +0.002 pts | Signal noyé sur 30 jours |
| Saisonnalité conditionnelle | 2.19% | Complexité sans gain |
| Ancrage de prévision | −0.06% | 3/12 folds améliorés seulement |
| TimesFM zero-shot | 8.3% | Battu par feature engineering |

---

## Licence

MIT
