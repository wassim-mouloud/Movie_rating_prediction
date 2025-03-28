import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path: str, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Charge les données depuis le fichier CSV, sépare le DataFrame en variables explicatives (X)
    et en cible (y), puis divise les données en ensemble d'entraînement et ensemble de test.
    
    Args:
        file_path (str): Chemin vers le fichier CSV contenant les données encodées.
        target (str): Nom de la colonne cible à prédire (ici 'Rating').
        test_size (float): Proportion de données à réserver pour l'ensemble de test (par défaut 0.2).
        random_state (int): Valeur de random state pour assurer la reproductibilité (par défaut 42).
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = "data/encoded_data.csv"  
    target = "Rating"
    X_train, X_test, y_train, y_test = split_data(file_path, target, test_size=0.2)
    
    print("Taille de l'ensemble d'entraînement :", X_train.shape)
    print("Taille de l'ensemble de test :", X_test.shape)
