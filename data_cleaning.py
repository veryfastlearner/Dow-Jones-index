import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# ============================================================
# 1. Chargement et nettoyage des données
# ============================================================
df = pd.read_csv("Dow Jones Industrial.csv", encoding="utf-8-sig")

# enlever les virgules de milliers et convertir en nombres
for c in ["Price", "Open", "High", "Low"]:
    df[c] = df[c].str.replace(",", "").astype(float)

# nettoyer Vol. (ex: "336.06M" -> 336060000, "1.25B" -> 1250000000)
def parse_volume(v):
    v = str(v).strip()
    if v.endswith("M"):
        return float(v[:-1]) * 1_000_000
    elif v.endswith("B"):
        return float(v[:-1]) * 1_000_000_000
    elif v.endswith("K"):
        return float(v[:-1]) * 1_000
    return float(v)

df["Vol."] = df["Vol."].apply(parse_volume)

# nettoyer Change % (ex: "-0.63%" -> -0.0063)
df["Change %"] = df["Change %"].str.replace("%", "").astype(float) / 100

# dates au format américain MM/JJ/AAAA, puis tri croissant
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)

# ============================================================
# 2. Construction des features et target
# ============================================================
Close = df["Price"]  # Price = Close (selon la remarque du professeur)
R = np.log(Close / Close.shift(1))          # R_t
x1 = R
x2 = (df["High"] - df["Low"]) / Close
y = R.shift(-1)                             # y_t = R_(t+1)

data = pd.DataFrame({
    "Date": df["Date"],
    "Price": df["Price"],
    "Open": df["Open"],
    "High": df["High"],
    "Low": df["Low"],
    "Vol.": df["Vol."],
    "Change %": df["Change %"],
    "x1": x1,
    "x2": x2,
    "y": y
}).dropna().reset_index(drop=True)

n = len(data)
print(f"Nombre d'observations utilisables (n = T-2): {n}")

# sauvegarde intermédiaire
data.to_csv("Dow_Jones_cleaned.csv", index=False)
print("Données nettoyées sauvegardées dans Dow_Jones_cleaned.csv")

# ============================================================
# 3. Découpage chronologique 80% / 20%
# ============================================================
split_idx = int(0.8 * n)
train = data.iloc[:split_idx].copy()
test = data.iloc[split_idx:].copy()

print(f"Entraînement: {len(train)} observations")
print(f"Test: {len(test)} observations")

# ============================================================
# Partie I : Régression linéaire et optimisation
# ============================================================
print("\n" + "="*60)
print("PARTIE I : RÉGRESSION LINÉAIRE")
print("="*60)

# Construction de X et y
# X : première colonne de 1, puis x1 et x2
X_train = np.column_stack([np.ones(len(train)), train["x1"].values, train["x2"].values])
y_train = train["y"].values
X_test = np.column_stack([np.ones(len(test)), test["x1"].values, test["x2"].values])
y_test = test["y"].values

print(f"\nDimensions : X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Dimensions : X_test = {X_test.shape}, y_test = {y_test.shape}")

# --- Question 2 : Développement de J(beta), gradient, hessienne ---
# J(beta) = (1/(2n)) * ||X beta - y||^2
#         = (1/(2n)) * (X beta - y)^T (X beta - y)
#         = (1/(2n)) * (beta^T X^T X beta - 2 y^T X beta + y^T y)
#
# Gradient : ∇J(beta) = (1/n) * X^T (X beta - y)
# Hessienne : ∇²J(beta) = (1/n) * X^T X
#
# Convexité : X^T X est semi-définie positive (pour tout v, v^T X^T X v = ||Xv||^2 >= 0)
#             Donc J est convexe. Si X est de rang plein, X^T X est définie positive
#             et J est strictement convexe.

def compute_J(X, y, beta):
    """Calcule la fonction coût J(beta) = (1/(2n)) * ||X beta - y||^2"""
    n = len(y)
    residuals = X @ beta - y
    return (1.0 / (2 * n)) * np.sum(residuals ** 2)

def compute_gradient(X, y, beta):
    """Calcule le gradient ∇J(beta) = (1/n) * X^T (X beta - y)"""
    n = len(y)
    return (1.0 / n) * X.T @ (X @ beta - y)

def compute_hessian(X):
    """Calcule la hessienne ∇²J(beta) = (1/n) * X^T X"""
    n = X.shape[0]
    return (1.0 / n) * X.T @ X

# Vérification de la convexité (valeurs propres de la hessienne)
H = compute_hessian(X_train)
eigenvals = np.linalg.eigvalsh(H)
print(f"\nValeurs propres de la hessienne (entraînement): {eigenvals}")
print(f"Toutes >= 0 ? {np.all(eigenvals >= -1e-10)} => J est convexe")
print(f"Toutes > 0 ? {np.all(eigenvals > 1e-10)} => J est strictement convexe (X de rang plein)")

# --- Question 4 & 5 : Calcul de beta_hat par équation normale et par Python ---
# Équation normale : X^T X beta_hat = X^T y  =>  beta_hat = (X^T X)^{-1} X^T y
beta_normal = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
print(f"\nCoefficients par équation normale :")
print(f"  beta_0 (intercept) = {beta_normal[0]:.10f}")
print(f"  beta_1 (x1)        = {beta_normal[1]:.10f}")
print(f"  beta_2 (x2)        = {beta_normal[2]:.10f}")

# Vérification de la condition d'optimalité : gradient = 0
grad_at_opt = compute_gradient(X_train, y_train, beta_normal)
print(f"\nGradient au point optimal (devrait être ~0): {grad_at_opt}")
print(f"Norme du gradient: {np.linalg.norm(grad_at_opt):.2e}")

# Comparaison avec une implémentation par descente de gradient (pour validation)
def gradient_descent(X, y, lr=0.01, max_iter=100000, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for i in range(max_iter):
        grad = compute_gradient(X, y, beta)
        beta_new = beta - lr * grad
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta, i + 1

beta_gd, n_iter = gradient_descent(X_train, y_train, lr=0.1)
print(f"\nCoefficients par descente de gradient ({n_iter} itérations):")
print(f"  beta_0 (intercept) = {beta_gd[0]:.10f}")
print(f"  beta_1 (x1)        = {beta_gd[1]:.10f}")
print(f"  beta_2 (x2)        = {beta_gd[2]:.10f}")
print(f"Différence avec équation normale: {np.linalg.norm(beta_normal - beta_gd):.2e}")

# --- Question 6 : MSE et R^2 sur entraînement et test ---
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# Prédictions
y_train_pred = X_train @ beta_normal
y_test_pred = X_test @ beta_normal

mse_train = compute_mse(y_train, y_train_pred)
mse_test = compute_mse(y_test, y_test_pred)
r2_train = compute_r2(y_train, y_train_pred)
r2_test = compute_r2(y_test, y_test_pred)

# Modèle de référence : prédire toujours la moyenne de y (sur l'entraînement)
y_mean_train = np.mean(y_train)
y_train_pred_baseline = np.full_like(y_train, y_mean_train)
y_test_pred_baseline = np.full_like(y_test, y_mean_train)

mse_train_baseline = compute_mse(y_train, y_train_pred_baseline)
mse_test_baseline = compute_mse(y_test, y_test_pred_baseline)
r2_train_baseline = compute_r2(y_train, y_train_pred_baseline)
r2_test_baseline = compute_r2(y_test, y_test_pred_baseline)

print(f"\n{'Métrique':<20} {'Entraînement':<15} {'Test':<15}")
print(f"{'-'*50}")
print(f"{'MSE (régression)':<20} {mse_train:<15.10f} {mse_test:<15.10f}")
print(f"{'R^2 (régression)':<20} {r2_train:<15.10f} {r2_test:<15.10f}")
print(f"{'MSE (baseline)':<20} {mse_train_baseline:<15.10f} {mse_test_baseline:<15.10f}")
print(f"{'R^2 (baseline)':<20} {r2_train_baseline:<15.10f} {r2_test_baseline:<15.10f}")

print(f"\nInterprétation :")
print(f"  - R^2 proche de 0 est attendu pour des rendements journaliers (marché efficace)")
print(f"  - Le modèle de régression explique très peu de la variance des rendements")
print(f"  - Signe de beta_1 (x1 = R_t) : {'positif' if beta_normal[1] > 0 else 'négatif'} (momentum {'fort' if abs(beta_normal[1]) > 0.1 else 'faible'})")
print(f"  - Signe de beta_2 (x2 = range/close) : {'positif' if beta_normal[2] > 0 else 'négatif'}")

# ============================================================
# Partie II : SVM pour classification hausse/baisse
# ============================================================
print("\n" + "="*60)
print("PARTIE II : SVM - CLASSIFICATION HAUSSE/BAISSE")
print("="*60)

# Étiquettes : z_t = +1 si R_{t+1} > 0, -1 sinon
z_train = np.where(train["y"] > 0, 1, -1)
z_test = np.where(test["y"] > 0, 1, -1)

print(f"\nRépartition des classes (entraînement) :")
print(f"  Hausse (+1) : {np.sum(z_train == 1)} ({100*np.mean(z_train == 1):.1f}%)")
print(f"  Baisse (-1) : {np.sum(z_train == -1)} ({100*np.mean(z_train == -1):.1f}%)")

# Standardisation des features (moyenne et écart-type sur l'entraînement uniquement)
x1_mean = train["x1"].mean()
x1_std = train["x1"].std()
x2_mean = train["x2"].mean()
x2_std = train["x2"].std()

X_train_std = np.column_stack([
    (train["x1"] - x1_mean) / x1_std,
    (train["x2"] - x2_mean) / x2_std
])
X_test_std = np.column_stack([
    (test["x1"] - x1_mean) / x1_std,
    (test["x2"] - x2_mean) / x2_std
])

# Entraînement SVM avec noyau linéaire
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_std, z_train)

# Paramètres de l'hyperplan : w1*x1 + w2*x2 + b = 0
w = svm.coef_[0]  # [w1, w2]
b = svm.intercept_[0]
w1, w2 = w[0], w[1]

print(f"\nParamètres de l'hyperplan affine (features standardisées) :")
print(f"  w1 = {w1:.6f}")
print(f"  w2 = {w2:.6f}")
print(f"  b  = {b:.6f}")
print(f"  Équation : {w1:.6f} * x1_std + {w2:.6f} * x2_std + {b:.6f} = 0")

# Prédictions et évaluation
z_train_pred = svm.predict(X_train_std)
z_test_pred = svm.predict(X_test_std)

acc_train = accuracy_score(z_train, z_train_pred)
acc_test = accuracy_score(z_test, z_test_pred)

print(f"\nPrécision (entraînement) : {acc_train:.4f}")
print(f"Précision (test)         : {acc_test:.4f}")
print(f"\nMatrice de confusion (test) :")
print(confusion_matrix(z_test, z_test_pred, labels=[1, -1]))

# Nombre de vecteurs de support
print(f"\nNombre de vecteurs de support : {len(svm.support_)}")
print(f"Indices des vecteurs de support : {svm.support_}")

# ============================================================
# Visualisations
# ============================================================
print("\nGénération des graphiques...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graphique 1 : Série temporelle des rendements
axes[0, 0].plot(data["Date"], data["y"], linewidth=0.5, alpha=0.7)
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_title("Rendements journaliers R_{t+1} (y)")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Rendement")
axes[0, 0].grid(True, alpha=0.3)

# Graphique 2 : nuage de points (x1, x2) coloré par y (régression)
scatter = axes[0, 1].scatter(data["x1"], data["x2"], c=data["y"], cmap='RdYlGn', alpha=0.6, s=10)
axes[0, 1].set_title("Nuage de points (x1, x2) coloré par y")
axes[0, 1].set_xlabel("x1 = R_t")
axes[0, 1].set_ylabel("x2 = (High - Low) / Close")
plt.colorbar(scatter, ax=axes[0, 1], label="y = R_{t+1}")
axes[0, 1].grid(True, alpha=0.3)

# Graphique 3 : Classes SVM (entraînement)
colors = ['red' if z == -1 else 'green' for z in z_train]
axes[1, 0].scatter(X_train_std[:, 0], X_train_std[:, 1], c=colors, alpha=0.6, s=10)
axes[1, 0].set_title("Classes SVM (entraînement, features standardisées)")
axes[1, 0].set_xlabel("x1 standardisé")
axes[1, 0].set_ylabel("x2 standardisé")
axes[1, 0].grid(True, alpha=0.3)

# Graphique 4 : Hyperplan SVM avec régions de décision
xx, yy = np.meshgrid(
    np.linspace(X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1, 200),
    np.linspace(X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1, 200)
)
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[-2, 0, 2])
axes[1, 1].scatter(X_train_std[:, 0], X_train_std[:, 1], c=colors, alpha=0.6, s=10, edgecolors='black', linewidth=0.5)

# Tracer l'hyperplan : w1*x + w2*y + b = 0 => y = -(w1*x + b) / w2
if abs(w2) > 1e-10:
    x_line = np.linspace(X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1, 100)
    y_line = -(w1 * x_line + b) / w2
    axes[1, 1].plot(x_line, y_line, 'k-', linewidth=2, label='Hyperplan SVM')

# Marge
margin = 1 / np.linalg.norm(w)
y_line_up = -(w1 * x_line + b - 1) / w2
y_line_down = -(w1 * x_line + b + 1) / w2
axes[1, 1].plot(x_line, y_line_up, 'k--', alpha=0.5, label='Marge')
axes[1, 1].plot(x_line, y_line_down, 'k--', alpha=0.5)

# Vecteurs de support
axes[1, 1].scatter(X_train_std[svm.support_, 0], X_train_std[svm.support_, 1],
                   s=100, facecolors='none', edgecolors='black', linewidth=1.5, label='Vecteurs de support')

axes[1, 1].set_title("SVM : Hyperplan et régions de décision (entraînement)")
axes[1, 0].set_xlabel("x1 standardisé")
axes[1, 0].set_ylabel("x2 standardisé")
axes[1, 1].legend(loc='best')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultats_optimisation.png", dpi=150, bbox_inches='tight')
print("Graphiques sauvegardés dans resultats_optimisation.png")

# ============================================================
# Résumé final
# ============================================================
print("\n" + "="*60)
print("RÉSUMÉ DES RÉSULTATS")
print("="*60)
print(f"\nPartie I - Régression linéaire :")
print(f"  beta_0 = {beta_normal[0]:.10f}")
print(f"  beta_1 = {beta_normal[1]:.10f}")
print(f"  beta_2 = {beta_normal[2]:.10f}")
print(f"  MSE train = {mse_train:.10f}, MSE test = {mse_test:.10f}")
print(f"  R^2 train = {r2_train:.6f}, R^2 test = {r2_test:.6f}")
print(f"\nPartie II - SVM :")
print(f"  w1 = {w1:.6f}, w2 = {w2:.6f}, b = {b:.6f}")
print(f"  Accuracy train = {acc_train:.4f}, Accuracy test = {acc_test:.4f}")
print(f"\nTous les résultats sont prêts pour le rapport PDF.")