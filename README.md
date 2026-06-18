# Prévision consommation électrique France, modèle Prophet

Modèle de prévision de la consommation électrique nationale française à horizons **7 jours** et **30 jours**, construit uniquement avec des données publiques gratuites.

> Projet initialisé dans le cadre d'une formation Data Engineer (Campus Numérique in the Alps) : prédiction de séries temporelles, déploiement de modèles de Machine Learning en production.

![Prévisions 7j sur 5 semaines représentatives de 2026](figures/readme_preview.png)

---

## Résultats

| Modèle | MAPE holdout (8 sem.) | MAPE test (2026) |
|--------|----------------------|------------------|
| **Prophet 7j** | **3.09%**[1] | **2.93%**[2] |
| **Prophet 30j** | **2.78%**[1] | **2.64%**[2] |
| Naïf saisonnier | - | 8.9%[3] |
| TimesFM zero-shot (200M) | - | 8.3%[3] |

[1] MAPE holdout : évaluation sur les 8 dernières semaines avant chaque réentraînement. Valeur non biaisée par la sélection des hyperparamètres. Un test Mann-Whitney vérifie l'absence d'overfitting à chaque retrain.

[2] MAPE de production : rolling hebdomadaire sur données 2026 jamais vues, avec réentraînement automatique bi-hebdomadaire.

[3] MAPE sur une seule fenêtre de test de 30 jours, protocole moins rigoureux que pour Prophet, à considérer comme indicatif. L'écart important avec Prophet (même peu optimisé, ~4% dès l'ajout de la température brute) a motivé la poursuite avec Prophet plutôt que TimesFM.

---

## Méthodologie

### Données
- **Consommation** : [RTE eco2mix](https://www.services-rte.com/fr/telechargez-les-donnees-publiees-par-rte.html) pour l'historique ("Courbe de consommation", foramt XLS, 15 minutes, agrégées en journalier), complété automatiquement par l'[API RTE Open Data](https://data.rte-france.com) pour les données récentes. Un fichier `rte_clean.csv` sert de data lake local.
- **Météo** : [Open-Meteo](https://open-meteo.com) : Archive API pour l'historique, Forecast API pour J+1 à J+16, moyenne climatologique (3 ans glissants) pour J+17 à J+30.

### Représentation spatiale de la température
4 points ruraux pondérés (hors îlots de chaleur urbains qui biaisent le modèle) :

| Point | Poids | Rôle |
|-------|-------|------|
| Alençon (NW) | 35% | Climat océanique, fort chauffage élec. |
| Bar-le-Duc (NE) | 30% | Climat continental, hivers froids |
| Périgueux (SW) | 20% | Climat tempéré |
| Montélimar (SE) | 15% | Pré-méditerranéen |

La pondération a été optimisée par cross-validation, aucune combinaison testée n'a fait mieux.

### Optimisation
- Hyperparamètres optimisés par **Optuna TPE** (optimisation bayésienne, 100 trials)
- Cross-validation Prophet : `initial='730 days'`, `period='30 days'`

![Importance des paramètres Optuna](figures/optuna_importance.png)

Sur les 100 trials, `heat_base_mean` et `fourier_yearly` sont les paramètres qui font le plus varier la MAPE. Les valeurs optimales sont dans `src/features.py`.

---

## Usage

```bash
# Run bi-hebdomadaire complet (validation + prévisions + retrain)
python run_weekly.py

# Réentraînement manuel
python retrain.py

# Réentraînement sans promotion en Production (mode simulation)
python retrain.py --dry-run

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
│   ├── data.py              # Chargement RTE + Open-Meteo, validation températures
│   ├── features.py          # HDD/CDD/lags/pct_vacances, BEST_PARAMS_7J et 30J
│   ├── model.py             # Build/train/predict Prophet 7j et 30j
│   └── evaluate.py          # Métriques, plots, test overfitting Mann-Whitney
├── retrain.py               # Réentraînement + évaluation holdout + promotion MLflow
├── run_weekly.py            # Pipeline bi-hebdomadaire (validation + prévisions + retrain)
├── run_if_needed.sh         # Guard idempotence pour le cron
├── cron_setup.sh            # Configuration automatique du cron
├── app.py                   # Dashboard Streamlit
├── requirements.txt
└── data/                    # Non versionné (.gitignore)
    ├── forecasts/                        # Prévisions datées (auto-généré)
    │   ├── forecast_7j_YYYY-MM-DD.csv
    │   └── forecast_30j_YYYY-MM-DD.csv
    ├── forecast_7j_latest.csv            # Dernière prévision 7j (auto-généré)
    ├── forecast_30j_latest.csv           # Dernière prévision 30j (auto-généré)
    ├── validation_log.csv                # Journal historique de validation (auto-généré)
    ├── temperature_forecast.csv          # Prévisions météo J+30 pour le dashboard (auto-généré)
    ├── conso_mix_RTE_YYYY.xls            # À télécharger depuis RTE
    ├── temp_Alencon.csv                  # Caches Open-Meteo par point (auto-généré)
    ├── temp_Bar_le_Duc.csv
    ├── temp_Perigueux.csv
    └── temp_Montelimar.csv
```

---

## Pistes abandonnées

| Piste | Résultat | Raison |
|-------|---------|--------|
| 8 grandes villes pondérées | +2 pts MAPE | Biais îlots de chaleur urbains |
| Fenêtre 2007-2025 | 5.93% | Concept drift (COVID, crise énergie 2022) |
| Poids asymétriques hiver/été | 2.47% | Pas mieux que symétrique |
| Vacances scolaires (30j) | +0.002 pts | Signal noyé sur 30 jours |
| Saisonnalité conditionnelle | 2.19% | Complexité sans gain |
| Ancrage de prévision | −0.06% | 3/12 prévisions améliorées seulement |
| TimesFM zero-shot | 8.3% | Battu par feature engineering |

---

## Licence

MIT
