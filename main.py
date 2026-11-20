"""
Script principal d'orchestration du projet d'optimisation.

Ce script coordonne l'ensemble du pipeline :
1. Chargement et nettoyage des données
2. Feature engineering
3. Découpage train/test
4. Partie I : Régression linéaire
5. Partie II : SVM
6. Génération des visualisations

Auteur : Projet d'optimisation - Analyse de données financières
"""

import matplotlib.pyplot as plt
import numpy as np

# Import des modules du projet
from data_loader import load_and_clean_data, save_cleaned_data
from feature_engineering import compute_features, get_feature_summary
from train_test_split import split_chronological, build_matrices, print_split_info
from linear_regression import run_linear_regression
from svm_classifier import run_svm_analysis


def plot_time_series(data, output_path="timeseries.png"):
    """
    Génère le graphique de la série temporelle des rendements.
    
    Args:
        data: DataFrame avec les features
        output_path: Chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["Date"], data["y"], linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title("Rendements journaliers R_{t+1} (y)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans {output_path}")


def plot_scatter_features(data, output_path="scatter_features.png"):
    """
    Génère le nuage de points (x1, x2) coloré par y.
    
    Args:
        data: DataFrame avec les features
        output_path: Chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data["x1"], data["x2"], c=data["y"], cmap='RdYlGn', alpha=0.6, s=10)
    ax.set_title("Nuage de points (x1, x2) coloré par y")
    ax.set_xlabel("x1 = R_t")
    ax.set_ylabel("x2 = (High - Low) / Close")
    plt.colorbar(scatter, ax=ax, label="y = R_{t+1}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé dans {output_path}")


def print_summary(reg_results, svm_results):
    """
    Affiche le résumé final des résultats.
    
    Args:
        reg_results: Résultats de la régression linéaire
        svm_results: Résultats du SVM
    """
    print("\n" + "="*60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*60)
    
    print(f"\nPartie I - Régression linéaire :")
    print(f"  beta_0 = {reg_results['beta'][0]:.10f}")
    print(f"  beta_1 = {reg_results['beta'][1]:.10f}")
    print(f"  beta_2 = {reg_results['beta'][2]:.10f}")
    print(f"  MSE train = {reg_results['mse_train']:.10f}, MSE test = {reg_results['mse_test']:.10f}")
    print(f"  R^2 train = {reg_results['r2_train']:.6f}, R^2 test = {reg_results['r2_test']:.6f}")
    
    print(f"\nPartie II - SVM :")
    print(f"  w1 = {svm_results['w1']:.6f}, w2 = {svm_results['w2']:.6f}, b = {svm_results['b']:.6f}")
    print(f"  Accuracy train = {svm_results['acc_train']:.4f}, Accuracy test = {svm_results['acc_test']:.4f}")
    
    print(f"\nTous les résultats sont prêts pour le rapport PDF.")


def main():
    """
    Fonction principale qui orchestre tout le pipeline.
    """
    print("="*60)
    print("PROJET D'OPTIMISATION - ANALYSE DE DONNÉES FINANCIÈRES")
    print("="*60)
    
    # 1. Chargement et nettoyage des données
    print("\n1. Chargement et nettoyage des données...")
    df = load_and_clean_data("Dow Jones Industrial.csv")
    print(f"   Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # 2. Feature engineering
    print("\n2. Feature engineering...")
    data = compute_features(df)
    print(f"   Features calculées : {len(data)} observations utilisables")
    get_feature_summary(data)
    
    # Sauvegarde des données nettoyées
    save_cleaned_data(data, "Dow_Jones_cleaned.csv")
    
    # 3. Découpage train/test
    print("\n3. Découpage chronologique 80% / 20%...")
    train, test = split_chronological(data)
    print_split_info(train, test)
    
    # Construction des matrices
    X_train, y_train, X_test, y_test = build_matrices(train, test)
    
    # 4. Partie I : Régression linéaire
    print("\n4. Partie I : Régression linéaire...")
    reg_results = run_linear_regression(X_train, y_train, X_test, y_test)
    
    # 5. Partie II : SVM
    print("\n5. Partie II : SVM...")
    svm_results = run_svm_analysis(train, test)
    
    # 6. Visualisations supplémentaires
    print("\n6. Génération des visualisations...")
    plot_time_series(data, "timeseries.png")
    plot_scatter_features(data, "scatter_features.png")
    
    # Résumé final
    print_summary(reg_results, svm_results)
    
    print("\n" + "="*60)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("="*60)


if __name__ == "__main__":
    main()