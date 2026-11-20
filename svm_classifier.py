"""
Module de classification SVM pour la prédiction hausse/baisse.

Partie II du projet : Support Vector Machine (SVM)

Étiquette :
    z_t = +1 si R_{t+1} > 0 (hausse)
    z_t = -1 si R_{t+1} <= 0 (baisse)

On garde les variables x_{1,t} et x_{2,t} de la partie I, que l'on standardise
(moyenne et écart-type calculés sur l'entraînement).

Problème d'optimisation :
    min_{w,b,xi} (1/2) ||w||^2 + C * sum(xi_i)
    sous z_i (w^T x_i + b) >= 1 - xi_i, xi_i >= 0

Hyperplan affine : w_1 * x_1 + w_2 * x_2 + b = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def create_labels(y):
    """
    Crée les étiquettes binaires pour la classification.
    
    Args:
        y: Vecteur des rendements futurs
        
    Returns:
        np.ndarray: Étiquettes +1 (hausse) ou -1 (baisse)
    """
    return np.where(y > 0, 1, -1)


def standardize_features(train, test):
    """
    Standardise les features x1 et x2.
    
    La moyenne et l'écart-type sont calculés sur l'entraînement uniquement,
    puis appliqués à l'entraînement et au test.
    
    Args:
        train: DataFrame d'entraînement
        test: DataFrame de test
        
    Returns:
        tuple: (X_train_std, X_test_std, mean_x1, std_x1, mean_x2, std_x2)
    """
    # Paramètres de standardisation (sur l'entraînement uniquement)
    x1_mean = train["x1"].mean()
    x1_std = train["x1"].std()
    x2_mean = train["x2"].mean()
    x2_std = train["x2"].std()
    
    # Standardisation
    X_train_std = np.column_stack([
        (train["x1"] - x1_mean) / x1_std,
        (train["x2"] - x2_mean) / x2_std
    ])
    X_test_std = np.column_stack([
        (test["x1"] - x1_mean) / x1_std,
        (test["x2"] - x2_mean) / x2_std
    ])
    
    return X_train_std, X_test_std, x1_mean, x1_std, x2_mean, x2_std


def train_svm(X_train, z_train, C=1.0):
    """
    Entraîne un SVM avec noyau linéaire.
    
    Args:
        X_train: Features standardisées d'entraînement
        z_train: Étiquettes d'entraînement
        C: Paramètre de régularisation
        
    Returns:
        SVC: Modèle SVM entraîné
    """
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train, z_train)
    return svm


def get_hyperplane_params(svm):
    """
    Extrait les paramètres de l'hyperplan affine.
    
    Args:
        svm: Modèle SVM entraîné
        
    Returns:
        tuple: (w1, w2, b)
    """
    w = svm.coef_[0]  # [w1, w2]
    b = svm.intercept_[0]
    return w[0], w[1], b


def evaluate_svm(svm, X_train, z_train, X_test, z_test):
    """
    Évalue les performances du SVM.
    
    Args:
        svm: Modèle SVM entraîné
        X_train: Features d'entraînement
        z_train: Étiquettes d'entraînement
        X_test: Features de test
        z_test: Étiquettes de test
        
    Returns:
        dict: Métriques d'évaluation
    """
    z_train_pred = svm.predict(X_train)
    z_test_pred = svm.predict(X_test)
    
    acc_train = accuracy_score(z_train, z_train_pred)
    acc_test = accuracy_score(z_test, z_test_pred)
    
    cm_test = confusion_matrix(z_test, z_test_pred, labels=[1, -1])
    
    return {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "confusion_matrix_test": cm_test,
        "n_support_vectors": len(svm.support_),
        "support_indices": svm.support_
    }


def plot_svm_results(X_train, z_train, svm, w1, w2, b, output_path="svm_results.png"):
    """
    Génère les graphiques des résultats SVM.
    
    Args:
        X_train: Features standardisées d'entraînement
        z_train: Étiquettes d'entraînement
        svm: Modèle SVM entraîné
        w1, w2, b: Paramètres de l'hyperplan
        output_path: Chemin de sauvegarde
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1 : Classes SVM
    colors = ['red' if z == -1 else 'green' for z in z_train]
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.6, s=10)
    axes[0].set_title("Classes SVM (entraînement, features standardisées)")
    axes[0].set_xlabel("x1 standardisé")
    axes[0].set_ylabel("x2 standardisé")
    axes[0].grid(True, alpha=0.3)
    
    # Graphique 2 : Hyperplan SVM avec régions de décision
    xx, yy = np.meshgrid(
        np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 200),
        np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 200)
    )
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[-2, 0, 2])
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.6, s=10, edgecolors='black', linewidth=0.5)
    
    # Tracer l'hyperplan : w1*x + w2*y + b = 0 => y = -(w1*x + b) / w2
    if abs(w2) > 1e-10:
        x_line = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
        y_line = -(w1 * x_line + b) / w2
        axes[1].plot(x_line, y_line, 'k-', linewidth=2, label='Hyperplan SVM')
        
        # Marge
        y_line_up = -(w1 * x_line + b - 1) / w2
        y_line_down = -(w1 * x_line + b + 1) / w2
        axes[1].plot(x_line, y_line_up, 'k--', alpha=0.5, label='Marge')
        axes[1].plot(x_line, y_line_down, 'k--', alpha=0.5)
    
    # Vecteurs de support
    axes[1].scatter(X_train[svm.support_, 0], X_train[svm.support_, 1],
                    s=100, facecolors='none', edgecolors='black', linewidth=1.5, label='Vecteurs de support')
    
    axes[1].set_title("SVM : Hyperplan et régions de décision (entraînement)")
    axes[1].set_xlabel("x1 standardisé")
    axes[1].set_ylabel("x2 standardisé")
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphiques SVM sauvegardés dans {output_path}")


def run_svm_analysis(train, test):
    """
    Exécute l'analyse complète SVM.
    
    Args:
        train: DataFrame d'entraînement
        test: DataFrame de test
        
    Returns:
        dict: Résultats de l'analyse
    """
    print("\n" + "="*60)
    print("PARTIE II : SVM - CLASSIFICATION HAUSSE/BAISSE")
    print("="*60)
    
    # Création des étiquettes
    z_train = create_labels(train["y"].values)
    z_test = create_labels(test["y"].values)
    
    print(f"\nRépartition des classes (entraînement) :")
    print(f"  Hausse (+1) : {np.sum(z_train == 1)} ({100*np.mean(z_train == 1):.1f}%)")
    print(f"  Baisse (-1) : {np.sum(z_train == -1)} ({100*np.mean(z_train == -1):.1f}%)")
    
    # Standardisation
    X_train_std, X_test_std, x1_mean, x1_std, x2_mean, x2_std = standardize_features(train, test)
    
    # Entraînement SVM
    svm = train_svm(X_train_std, z_train)
    
    # Paramètres de l'hyperplan
    w1, w2, b = get_hyperplane_params(svm)
    
    print(f"\nParamètres de l'hyperplan affine (features standardisées) :")
    print(f"  w1 = {w1:.6f}")
    print(f"  w2 = {w2:.6f}")
    print(f"  b  = {b:.6f}")
    print(f"  Équation : {w1:.6f} * x1_std + {w2:.6f} * x2_std + {b:.6f} = 0")
    
    # Évaluation
    metrics = evaluate_svm(svm, X_train_std, z_train, X_test_std, z_test)
    
    print(f"\nPrécision (entraînement) : {metrics['acc_train']:.4f}")
    print(f"Précision (test)         : {metrics['acc_test']:.4f}")
    print(f"\nMatrice de confusion (test) :")
    print(metrics['confusion_matrix_test'])
    print(f"\nNombre de vecteurs de support : {metrics['n_support_vectors']}")
    
    # Graphiques
    plot_svm_results(X_train_std, z_train, svm, w1, w2, b)
    
    return {
        "w1": w1,
        "w2": w2,
        "b": b,
        "acc_train": metrics['acc_train'],
        "acc_test": metrics['acc_test'],
        "confusion_matrix_test": metrics['confusion_matrix_test'],
        "n_support_vectors": metrics['n_support_vectors'],
        "svm": svm,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "z_train": z_train,
        "z_test": z_test
    }


if __name__ == "__main__":
    # Test du module
    from data_loader import load_and_clean_data
    from feature_engineering import compute_features
    from train_test_split import split_chronological
    
    df = load_and_clean_data()
    data = compute_features(df)
    train, test = split_chronological(data)
    
    results = run_svm_analysis(train, test)