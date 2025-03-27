# src/main.py
from data_preprocessing import load_data, explore_data
from data_cleaning import clean_data
import pandas as pd

def main():
    file_path = "data/raw_data.csv"
    
    print("Chargement des données...")
    df = load_data(file_path)
    
    print("\nExploration initiale des données brutes :")
    explore_data(df)
    
    print("\nNettoyage des données...")
    df_clean = clean_data(df)
    
    print("\nExploration des données nettoyées :")
    explore_data(df_clean)
    
    cleaned_file_path = "data/clean_data.csv"
    df_clean.to_csv(cleaned_file_path, index=False)
    print(f"\nDonnées nettoyées sauvegardées dans : {cleaned_file_path}")

if __name__ == "__main__":
    main()
