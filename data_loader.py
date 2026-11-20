"""
Module de chargement et nettoyage des données financières.

Ce module gère :
- Le chargement du fichier CSV Dow Jones Industrial
- Le nettoyage des colonnes numériques (virgules de milliers, suffixes M/B/K)
- La conversion des dates au format MM/JJ/AAAA
- Le tri chronologique des données
"""

import pandas as pd
import numpy as np


def parse_volume(v):
    """
    Convertit une chaîne de volume avec suffixe en nombre entier.
    
    Exemples :
        "336.06M" -> 336060000
        "1.25B"   -> 1250000000
        "500K"    -> 500000
        "1000"    -> 1000
    
    Args:
        v: Chaîne de caractères représentant un volume
        
    Returns:
        float: Volume en nombre entier
    """
    v = str(v).strip()
    if v.endswith("M"):
        return float(v[:-1]) * 1_000_000
    elif v.endswith("B"):
        return float(v[:-1]) * 1_000_000_000
    elif v.endswith("K"):
        return float(v[:-1]) * 1_000
    return float(v)


def load_and_clean_data(filepath="Dow Jones Industrial.csv"):
    """
    Charge et nettoie les données du Dow Jones Industrial.
    
    Étapes :
    1. Lecture du CSV avec encodage utf-8-sig (gestion du BOM)
    2. Nettoyage des colonnes Price, Open, High, Low (suppression des virgules)
    3. Parsing de la colonne Vol. avec suffixes M/B/K
    4. Conversion de Change % (suppression du %, division par 100)
    5. Conversion des dates au format MM/JJ/AAAA
    6. Tri chronologique croissant
    
    Args:
        filepath: Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame nettoyé et trié par date croissante
    """
    # Lecture avec encodage utf-8-sig pour gérer le BOM
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    
    # Nettoyage des colonnes numériques (suppression des virgules de milliers)
    for c in ["Price", "Open", "High", "Low"]:
        df[c] = df[c].str.replace(",", "").astype(float)
    
    # Nettoyage de la colonne Vol. (ex: "336.06M" -> 336060000)
    df["Vol."] = df["Vol."].apply(parse_volume)
    
    # Nettoyage de Change % (ex: "-0.63%" -> -0.0063)
    df["Change %"] = df["Change %"].str.replace("%", "").astype(float) / 100
    
    # Conversion des dates au format américain MM/JJ/AAAA
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    
    # Tri chronologique croissant
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df


def save_cleaned_data(df, output_path="Dow_Jones_cleaned.csv"):
    """
    Sauvegarde les données nettoyées dans un fichier CSV.
    
    Args:
        df: DataFrame nettoyé
        output_path: Chemin de sortie
    """
    df.to_csv(output_path, index=False)
    print(f"Données nettoyées sauvegardées dans {output_path}")


if __name__ == "__main__":
    # Test du module
    df = load_and_clean_data()
    print(f"Dimensions : {df.shape}")
    print(f"\nColonnes : {list(df.columns)}")
    print(f"\nPremières lignes :")
    print(df.head())
    print(f"\nDernières lignes :")
    print(df.tail())