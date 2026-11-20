"""
Module de régression linéaire pour la prédiction des rendements.

Partie I du projet : Régression linéaire et optimisation

Modèle :
    y_hat_t = beta_0 + beta_1 * x_{1,t} + beta_2 * x_{2,t}

Problème d'optimisation :
    min_{beta} J(beta) = (1/(2n)) * ||X beta - y||^2

Développements mathématiques :
    J(beta) = (1/(2n)) * (X beta - y)^T (X beta - y)
            = (1/(2n)) * (beta^T X^T X beta - 2 y^T X beta + y^T y)

    Gradient : ∇J(beta) = (1/n) * X^T (X beta - y)
    Hessienne : ∇²J(beta) = (1/n) * X^T X

    Convexité : X^T X est semi-définie positive (v^T X^T X v = ||Xv||^2 >= 0)
                Donc J est convexe. Si X est de rang plein, J est strictement convexe.

    Condition d'optimalité : ∇J(beta_hat) = 0
    Équation normale : X^T X beta_hat = X^T y
                       => beta_hat = (X^T X)^{-1} X^T y
"""

import numpy as np


def compute_J(X, y, beta):
    """
    Calcule la fonction coût J(beta) = (1/(2n)) * ||X beta - y||^2.
    
    Args:
        X: Matrice de features (n x 3)
        y: Vecteur cible (n,)
        beta: Vecteur de coefficients (3,)
        
    Returns:
        float: Valeur de la fonction coût
    """
    n = len(y)
    residuals = X @ beta - y
    return (1.0 / (2 * n)) * np.sum(residuals ** 2)


def compute_gradient(X, y, beta):
    """
    Calcule le gradient ∇J(beta) = (1/n) * X^T (X beta - y).
    
    Args:
        X: Matrice de features (n x 3)
        y: Vecteur cible (n,)
        beta: Vecteur de coefficients (3,)
        
    Returns:
        np.ndarray: Gradient (3,)
    """
    n = len(y)
    return (1.0 / n) * X.T @ (X @ beta - y)


def compute_hessian(X):
    """
    Calcule la hessienne ∇²J(beta) = (1/n) * X^T X.
    
    Args:
        X: Matrice de features (n x 3)
        
    Returns:
        np.ndarray: Hessienne (3 x 3)
    """
    n = X.shape[0]
    return (1.0 / n) * X.T @ X


def check_convexity(X):
    """
    Vérifie la convexité de J en calculant les valeurs propres de la hessienne.
    
    Args:
        X: Matrice de features
        
    Returns:
        tuple: (eigenvalues, is_convex, is_strictly_convex)
    """
    H = compute_hessian(X)
    eigenvalues = np.linalg.eigvalsh(H)
    
    is_convex = np.all(eigenvalues >= -1e-10)
    is_strictly_convex = np.all(eigenvalues > 1e-10)
    
    return eigenvalues, is_convex, is_strictly_convex


def solve_normal_equation(X, y):
    """
    Résout l'équation normale X^T X beta = X^T y.
    
    Args:
        X: Matrice de features (n x 3)
        y: Vecteur cible (n,)
        
    Returns:
        np.ndarray: Coefficients beta_hat (3,)
    """
    return np.linalg.solve(X.T @ X, X.T @ y)


def gradient_descent(X, y, lr=0.1, max_iter=100000, tol=1e-10):
    """
    Résout le problème par descente de gradient (pour validation).
    
    Args:
        X: Matrice de features
        y: Vecteur cible
        lr: Taux d'apprentissage
        max_iter: Nombre maximum d'itérations
        tol: Tolérance de convergence
        
    Returns:
        tuple: (beta, n_iterations)
    """
    beta = np.zeros(X.shape[1])
    for i in range(max_iter):
        grad = compute_gradient(X, y, beta)
        beta_new = beta - lr * grad
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta, i + 1


def compute_mse(y_true, y_pred):
    """
    Calcule l'erreur quadratique moyenne (MSE).
    
    Args:
        y_true: Valeurs vraies
        y_pred: Valeurs prédites
        
    Returns:
        float: MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    """
    Calcule le coefficient de détermination R^2.
    
    Args:
        y_true: Valeurs vraies
        y_pred: Valeurs prédites
        
    Returns:
        float: R^2
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def run_linear_regression(X_train, y_train, X_test, y_test):
    """
    Exécute l'analyse complète de régression linéaire.
    
    Args:
        X_train: Matrice d'entraînement
        y_train: Vecteur cible d'entraînement
        X_test: Matrice de test
        y_test: Vecteur cible de test
        
    Returns:
        dict: Résultats de l'analyse
    """
    print("\n" + "="*60)
    print("PARTIE I : RÉGRESSION LINÉAIRE")
    print("="*60)
    
    # Question 1 : Dimensions
    print(f"\nDimensions : X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"Dimensions : X_test = {X_test.shape}, y_test = {y_test.shape}")
    
    # Question 2 : Convexité
    eigenvalues, is_convex, is_strictly_convex = check_convexity(X_train)
    print(f"\nValeurs propres de la hessienne (entraînement): {eigenvalues}")
    print(f"Toutes >= 0 ? {is_convex} => J est convexe")
    print(f"Toutes > 0 ? {is_strictly_convex} => J est strictement convexe (X de rang plein)")
    
    # Question 4 & 5 : Coefficients par équation normale
    beta_normal = solve_normal_equation(X_train, y_train)
    print(f"\nCoefficients par équation normale :")
    print(f"  beta_0 (intercept) = {beta_normal[0]:.10f}")
    print(f"  beta_1 (x1)        = {beta_normal[1]:.10f}")
    print(f"  beta_2 (x2)        = {beta_normal[2]:.10f}")
    
    # Vérification de la condition d'optimalité
    grad_at_opt = compute_gradient(X_train, y_train, beta_normal)
    print(f"\nGradient au point optimal (devrait être ~0): {grad_at_opt}")
    print(f"Norme du gradient: {np.linalg.norm(grad_at_opt):.2e}")
    
    # Comparaison avec descente de gradient
    beta_gd, n_iter = gradient_descent(X_train, y_train, lr=0.1)
    print(f"\nCoefficients par descente de gradient ({n_iter} itérations):")
    print(f"  beta_0 (intercept) = {beta_gd[0]:.10f}")
    print(f"  beta_1 (x1)        = {beta_gd[1]:.10f}")
    print(f"  beta_2 (x2)        = {beta_gd[2]:.10f}")
    print(f"Différence avec équation normale: {np.linalg.norm(beta_normal - beta_gd):.2e}")
    
    # Question 6 : MSE et R^2
    y_train_pred = X_train @ beta_normal
    y_test_pred = X_test @ beta_normal
    
    mse_train = compute_mse(y_train, y_train_pred)
    mse_test = compute_mse(y_test, y_test_pred)
    r2_train = compute_r2(y_train, y_train_pred)
    r2_test = compute_r2(y_test, y_test_pred)
    
    # Modèle de référence : prédire toujours la moyenne de y
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
    
    # Question 7 : Interprétation
    print(f"\nInterprétation :")
    print(f"  - R^2 proche de 0 est attendu pour des rendements journaliers (marché efficace)")
    print(f"  - Le modèle de régression explique très peu de la variance des rendements")
    print(f"  - Signe de beta_1 (x1 = R_t) : {'positif' if beta_normal[1] > 0 else 'négatif'} (momentum {'fort' if abs(beta_normal[1]) > 0.1 else 'faible'})")
    print(f"  - Signe de beta_2 (x2 = range/close) : {'positif' if beta_normal[2] > 0 else 'négatif'}")
    
    return {
        "beta": beta_normal,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mse_train_baseline": mse_train_baseline,
        "mse_test_baseline": mse_test_baseline,
        "r2_train_baseline": r2_train_baseline,
        "r2_test_baseline": r2_test_baseline
    }


if __name__ == "__main__":
    # Test du module
    from data_loader import load_and_clean_data
    from feature_engineering import compute_features
    from train_test_split import split_chronological, build_matrices
    
    df = load_and_clean_data()
    data = compute_features(df)
    train, test = split_chronological(data)
    X_train, y_train, X_test, y_test = build_matrices(train, test)
    
    results = run_linear_regression(X_train, y_train, X_test, y_test)