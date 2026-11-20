# Projet : Analyse de données financières

## 1. Jeu de données et objectif

On travaille sur les données quotidiennes du **Dow Jones Industrial Average (DJIA)**, indice regroupant 30 grandes entreprises américaines, de **janvier 2015 à décembre 2025**.

Chaque ligne correspond à une séance boursière et contient : `Date`, `Open`, `High`, `Low`, `Close` et `Volume`.

- **Date** : date de la séance boursière.
- **Open** : valeur de l'indice à l'ouverture de la séance.
- **High** : valeur maximale atteinte durant la séance.
- **Low** : valeur minimale atteinte durant la séance.
- **Close** : valeur de l'indice à la clôture de la séance.
- **Volume** : volume des transactions associé à la série de marché.

> **Remarque du professeur :** dans le fichier `Dow_Jones_Industrial.csv`, la colonne **`Price`** correspond à **`Close`** (prix de clôture quotidien). Il faut donc utiliser `Price` pour représenter `Close` dans tous les calculs.

**NB :** dans ce projet, on utilise uniquement la colonne Close (donc `Price`), plus `High` et `Low` pour construire $x_2$.

### Objectif

Appliquer des méthodes d'optimisation à des données réelles :

1. **Partie I** : une régression linéaire pour prédire le rendement financier du lendemain.
2. **Partie II** : un SVM pour construire un hyperplan affine séparant deux classes : hausse et baisse du marché.

---

## 2. Préparation des données

Trier les données par **date croissante**, puis calculer, pour $t = 2, \dots, T-1$ ($n = T-2$ observations utilisables) :

$$R_t = \ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right), \qquad y_t = R_{t+1}, \qquad x_{1,t} = R_t, \qquad x_{2,t} = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t}$$

**Découpage chronologique :**

- **80 %** premières observations : entraînement.
- **20 %** dernières observations : test.
- **Pas de mélange aléatoire.**
- **Tous les paramètres sont estimés sur l'entraînement uniquement.**

---

## Partie I : Régression linéaire et optimisation

**But :** construire un modèle de régression linéaire qui prédit le rendement du lendemain. On cherche les coefficients qui minimisent l'écart entre rendements réels et prédits, mesuré par la somme des carrés des erreurs.

**Modèle :**

$$\hat y_t = \beta_0 + \beta_1 x_{1,t} + \beta_2 x_{2,t}$$

**Problème d'optimisation :**

$$\min_{\beta \in \mathbb{R}^3} J(\beta) = \frac{1}{2n}\,\|X\beta - y\|_2^2$$

où $X \in \mathbb{R}^{n \times 3}$ (première colonne de 1, puis $x_1$ et $x_2$), $y \in \mathbb{R}^n$ et $\beta = (\beta_0, \beta_1, \beta_2)^T$.

### Travail demandé

1. **Construire $X$ et $y$** (donner leurs dimensions).
2. **Développer $J(\beta)$**, calculer son **gradient** $\nabla J(\beta)$ et sa **hessienne** $\nabla^2 J(\beta)$. **Montrer que $J$ est convexe.**
3. **Écrire la condition d'optimalité** $\nabla J(\hat\beta) = 0$ et en déduire l'**équation normale** $X^T X \hat\beta = X^T y$.
4. **Calculer $\hat\beta$** en résolvant l'équation normale.
5. **Utiliser Python** pour déterminer les coefficients optimaux de la régression linéaire, et **comparer** avec ceux obtenus par l'équation normale (question 4).
6. **Calculer** les rendements prédits, le **MSE** et le coefficient de détermination **$R^2$**, sur l'entraînement **et** sur le test. **Comparer** avec le modèle qui prédit toujours la moyenne de $y$.
7. **Interpréter** les résultats : signe des coefficients, valeur de $R^2$, limites du modèle. *Un $R^2$ proche de 0 est attendu pour des rendements journaliers.*

---

## Partie II : Support Vector Machine (SVM)

**Étiquette :**

$$z_t = \begin{cases} +1 & \text{si } R_{t+1} > 0 \text{ (hausse)} \\ -1 & \text{si } R_{t+1} \le 0 \text{ (baisse)} \end{cases}$$

On garde les variables $x_{1,t}$ et $x_{2,t}$ de la partie I, que l'on **standardise** (moyenne et écart-type calculés sur l'entraînement).

On cherche un hyperplan affine $w_1 x_1 + w_2 x_2 + b = 0$ en résolvant :

$$\min_{w,b,\xi} \ \frac12 \|w\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{sous} \quad z_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

### Travail demandé

1. **Construire les deux classes** (hausse et baisse) et les **représenter dans le plan $(x_1, x_2)$** avec deux couleurs différentes.
2. **Construire l'hyperplan affine** (par exemple avec `sklearn.svm.SVC(kernel='linear')`) et **identifier ses paramètres** $w_1$, $w_2$ et $b$.
3. **Représenter graphiquement l'hyperplan et les deux régions de classification.** Interpréter la séparation obtenue et discuter les éventuelles erreurs de classification.

---

## Consignes générales et modalités de rendu

- Le travail est **individuel**.
- L'utilisation de **Python** est fortement recommandée (calculs numériques et graphiques).
- Le compte rendu doit présenter les **développements mathématiques**, les **résultats numériques** et leur **interprétation / analyse**.
- Les **graphiques** doivent être lisibles et comporter un **titre**, des **axes identifiés** et une **légende**.

### Document à rendre

Un **rapport au format PDF** présentant :

- Les développements et justifications mathématiques.
- Les résultats pratiques obtenus.
- Les tableaux et représentations graphiques nécessaires.
- L'interprétation et la discussion des résultats.

> **Remarque :** le code Python **n'est pas demandé**. Seuls les résultats, accompagnés de leurs explications et interprétations, doivent figurer dans le rapport.

**Données :** le fichier de données financières est fourni avec l'énoncé (`Dow_Jones_Industrial.csv`).

*Bon travail !*

---

## Checklist rapide

- [ ] Charger le CSV, nettoyer les nombres (virgules de milliers), convertir les dates (MM/JJ/AAAA)
- [ ] Trier par date croissante
- [ ] Utiliser `Price` comme `Close`
- [ ] Calculer $R_t$, $x_1$, $x_2$, $y$ (soit $n = T-2$ observations)
- [ ] Découper 80 % / 20 % chronologiquement
- [ ] Partie I, questions 1 à 7
- [ ] Partie II, questions 1 à 3 (standardisation avec les paramètres de l'entraînement)
- [ ] Rapport PDF : maths, résultats, tableaux, graphiques (titre, axes, légende), interprétation
