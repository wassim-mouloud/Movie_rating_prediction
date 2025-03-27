import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV contenant les données brutes.
    
    Args:
        file_path (str): Chemin vers le fichier CSV.
    
    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Fichier chargé avec succès : {file_path}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        raise

def explore_data(df: pd.DataFrame) -> None:
    """
    Affiche les informations de base sur le DataFrame pour l'exploration initiale.
    
    Args:
        df (pd.DataFrame): DataFrame à explorer.
    """
    print("Aperçu des 5 premières lignes :")
    print(df.head(20), "\n")
    
    # print("Informations sur le DataFrame :")
    # print(df.info(), "\n")
    
    # print("Statistiques descriptives :")
    # print(df.describe(), "\n")
    
    print("Dimensions du DataFrame : ", df.shape)
