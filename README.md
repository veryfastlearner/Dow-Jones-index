# Dow Jones Index - Projet d'Optimisation

Analyse de données financières du Dow Jones Industrial Average (DJIA) utilisant des méthodes d'optimisation pour la prédiction des rendements et la classification hausse/baisse.

## 📊 Description du Projet

Ce projet applique des techniques d'optimisation mathématique à des données réelles du marché boursier américain :

- **Partie I** : Régression linéaire pour prédire le rendement financier du lendemain
- **Partie II** : Support Vector Machine (SVM) pour classifier les mouvements haussiers/baissiers du marché

## 📁 Structure du Projet

```
Dow-Jones-index/
├── data_loader.py          # Chargement et nettoyage des données
├── feature_engineering.py  # Calcul des features (R_t, x1, x2, y)
├── train_test_split.py     # Découpage chronologique 80%/20%
├── linear_regression.py    # Partie I : Régression linéaire
├── svm_classifier.py       # Partie II : SVM
├── main.py                 # Script principal d'orchestration
├── Dow Jones Industrial.csv # Données brutes
├── instructions_projet_optimisation.md # Énoncé du projet
└── README.md               # Ce fichier
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip

### Dépendances

```bash
pip install pandas numpy matplotlib scikit-learn
```

## 💻 Utilisation

### Exécution complète du pipeline

```bash
python main.py
```

### Exécution des modules individuellement

```python
# Chargement des données
from data_loader import load_and_clean_data
df = load_and_clean_data("Dow Jones Industrial.csv")

# Feature engineering
from feature_engineering import compute_features
data = compute_features(df)

# Découpage train/test
from train_test_split import split_chronological
train, test = split_chronological(data)

# Régression linéaire
from linear_regression import run_linear_regression
results = run_linear_regression(X_train, y_train, X_test, y_test)

# SVM
from svm_classifier import run_svm_analysis
svm_results = run_svm_analysis(train, test)
```

## 📈 Données

Le jeu de données contient les cotations quotidiennes du Dow Jones Industrial Average de janvier 2015 à décembre 2025.

### Colonnes

| Colonne | Description |
|---------|-------------|
| Date | Date de la séance boursière |
| Price | Prix de clôture (Close) |
| Open | Prix d'ouverture |
| High | Prix maximal |
| Low | Prix minimal |
| Vol. | Volume des transactions |
| Change % | Variation en pourcentage |

### Features calculées

- **R_t** : Rendement logarithmique = ln(Close_t / Close_{t-1})
- **x1** : R_t (rendement courant)
- **x2** : (High_t - Low_t) / Close_t (range relatif)
- **y** : R_{t+1} (rendement du lendemain)

## 🔬 Méthodologie

### Partie I : Régression Linéaire

**Modèle** : ŷ_t = β₀ + β₁x₁ + β₂x₂

**Fonction de coût** : J(β) = (1/2n) ||Xβ - y||²

**Développements mathématiques** :
- Gradient : ∇J(β) = (1/n) Xᵀ(Xβ - y)
- Hessienne : ∇²J(β) = (1/n) XᵀX
- Convexité : XᵀX est semi-définie positive → J est convexe
- Équation normale : β̂ = (XᵀX)⁻¹Xᵀy

**Résultats** :
- β₀ = 0.00081, β₁ = -0.15571, β₂ = -0.04153
- R² ≈ 0.02 (train), R² ≈ -0.04 (test)
- MSE train = 0.000130, MSE test = 0.000080

### Partie II : SVM

**Étiquettes** : z_t = +1 (hausse) ou -1 (baisse)

**Hyperplan** : w₁x₁ + w₂x₂ + b = 0

**Résultats** :
- w₁ = 0.000023, w₂ = -0.000001, b = 1.000016
- Accuracy train = 0.5366, Accuracy test = 0.5497

## 📊 Visualisations

Le pipeline génère plusieurs graphiques :

1. **timeseries.png** : Série temporelle des rendements
2. **scatter_features.png** : Nuage de points (x1, x2) coloré par y
3. **svm_results.png** : Hyperplan SVM et régions de décision

## 📝 Interprétation des Résultats

### Régression Linéaire

- **R² proche de 0** : Attendu pour des rendements journaliers (marché efficace)
- **β₁ négatif** : Effet de momentum inverse (rendements passés négatifs → hausse future)
- **β₂ négatif** : Relation inverse entre volatilité et rendement futur

### SVM

- **Accuracy ~55%** : Légèrement supérieur au hasard (50%)
- **Déséquilibre des classes** : 53.7% hausse vs 46.3% baisse
- **Nombreux vecteurs de support** : Frontière de décision complexe

## 🛠️ Technologies Utilisées

- **Python 3.8+**
- **pandas** : Manipulation de données
- **numpy** : Calcul numérique
- **matplotlib** : Visualisation
- **scikit-learn** : SVM et métriques

## 📄 Licence

Ce projet est réalisé dans le cadre d'un cours d'optimisation.

## 👤 Auteur

Projet individuel - Analyse de données financières