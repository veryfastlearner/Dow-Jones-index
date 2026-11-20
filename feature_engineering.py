"""
Module de feature engineering pour les données financières.

Ce module calcule les variables nécessaires pour les modèles :
- R_t : rendement logarithmique à la date t
- x1 : variable explicative 1 (R_t)
- x2 : variable explicative 2 (range/close)
- y : variable cible (R_{t+1})

Formules :
    R_t = ln(Close_t / Close_{t-1})
    x_{1,t} = R_t
    x_{2,t} = (High_t - Low_t) / Close_t
    y_t = R_{t+1}

Note : Dans ce projet, la colonne "Price" correspond à "Close" (prix de clôture).
"""

import pandas as pd
import numpy as np


def compute_features(df):
    """
    Calcule les features et la variable cible à partir des données nettoyées.
    
    Args:
        df: DataFrame avec colonnes Date, Price, Open, High, Low, Vol., Change %
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes originales + x1, x2, y
    """
    # Price = Close (selon la remarque du professeur)
    Close = df["Price"]
    
    # R_t = ln(Close_t / Close_{t-1})
    R = np.log(Close / Close.shift(1))
    
    # x1 = R_t
    x1 = R
    
    # x2 = (High_t - Low_t) / Close_t
    x2 = (df["High"] - df["Low"]) / Close
    
    # y = R_{t+1} (rendement du lendemain)
    y = R.shift(-1)
    
    # Construction du DataFrame avec toutes les colonnes
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
    })
    
    # Suppression des lignes avec NaN (première et dernière ligne)
    data = data.dropna().reset_index(drop=True)
    
    return data


def get_feature_summary(data):
    """
    Affiche un résumé des features calculées.
    
    Args:
        data: DataFrame avec les features
    """
    n = len(data)
    print(f"Nombre d'observations utilisables (n = T-2) : {n}")
    print(f"\nStatistiques descriptives :")
    print(data[["x1", "x2", "y"]].describe())
    
    print(f"\nCorrélations :")
    print(data[["x1", "x2", "y"]].corr())


if __name__ == "__main__":
    # Test du module
    from data_loader import load_and_clean_data
    
    df = load_and_clean_data()
    data = compute_features(df)
    get_feature_summary(data)
    
    print(f"\nPremières lignes avec features :")
    print(data.head())