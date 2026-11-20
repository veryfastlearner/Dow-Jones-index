"""
Module de découpage chronologique des données.

Ce module gère le découpage train/test selon les spécifications :
- 80% premières observations : entraînement
- 20% dernières observations : test
- Pas de mélange aléatoire (découpage chronologique)
- Tous les paramètres sont estimés sur l'entraînement uniquement
"""

import pandas as pd
import numpy as np


def split_chronological(data, train_ratio=0.8):
    """
    Découpe les données de manière chronologique en train/test.
    
    Args:
        data: DataFrame avec les features et target
        train_ratio: Proportion pour l'entraînement (défaut: 0.8)
        
    Returns:
        tuple: (train, test) - deux DataFrames
    """
    n = len(data)
    split_idx = int(train_ratio * n)
    
    train = data.iloc[:split_idx].copy().reset_index(drop=True)
    test = data.iloc[split_idx:].copy().reset_index(drop=True)
    
    return train, test


def build_matrices(train, test):
    """
    Construit les matrices X et vecteurs y pour la régression linéaire.
    
    X : première colonne de 1, puis x1 et x2
    y : variable cible
    
    Args:
        train: DataFrame d'entraînement
        test: DataFrame de test
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    # Construction de X : [1, x1, x2]
    X_train = np.column_stack([
        np.ones(len(train)),
        train["x1"].values,
        train["x2"].values
    ])
    y_train = train["y"].values
    
    X_test = np.column_stack([
        np.ones(len(test)),
        test["x1"].values,
        test["x2"].values
    ])
    y_test = test["y"].values
    
    return X_train, y_train, X_test, y_test


def print_split_info(train, test):
    """
    Affiche les informations sur le découpage.
    
    Args:
        train: DataFrame d'entraînement
        test: DataFrame de test
    """
    print(f"Découpage chronologique 80% / 20% :")
    print(f"  Entraînement : {len(train)} observations")
    print(f"  Test         : {len(test)} observations")
    print(f"  Total        : {len(train) + len(test)} observations")
    
    print(f"\nPériode d'entraînement : {train['Date'].min()} à {train['Date'].max()}")
    print(f"Période de test        : {test['Date'].min()} à {test['Date'].max()}")


if __name__ == "__main__":
    # Test du module
    from data_loader import load_and_clean_data
    from feature_engineering import compute_features
    
    df = load_and_clean_data()
    data = compute_features(df)
    
    train, test = split_chronological(data)
    print_split_info(train, test)
    
    X_train, y_train, X_test, y_test = build_matrices(train, test)
    print(f"\nDimensions des matrices :")
    print(f"  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_test  : {y_test.shape}")