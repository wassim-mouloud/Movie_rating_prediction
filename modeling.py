import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
import joblib

from data_preprocessing import load_data, explore_data
from data_split import split_data

def train_linear_regression(X_train, y_train, poly_degree=2):
    """
    Entraîne un modèle de régression linéaire sur l'ensemble d'entraînement en ajoutant
    une transformation polynomiale pour capturer des relations non linéaires.

    Args:
        X_train (pd.DataFrame): Variables explicatives d'entraînement.
        y_train (pd.Series): Variable cible d'entraînement.
        poly_degree (int): Degré de la transformation polynomiale (par défaut 2).

    Returns:
        Pipeline: Pipeline entraîné comprenant standardisation, transformation polynomiale et modèle linéaire.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ('linreg', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    print("Modèle de régression linéaire entraîné avec transformation polynomiale (degré {})".format(poly_degree))
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test et affiche RMSE et R².

    Args:
        model (Pipeline): Pipeline entraîné.
        X_test (pd.DataFrame): Variables explicatives de test.
        y_test (pd.Series): Variable cible de test.

    Returns:
        tuple: (RMSE, R²)
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Évaluation du modèle :\nRMSE = {rmse:.3f}\nR² = {r2:.3f}")
    return rmse, r2

def cross_validate_model(model, X, y, cv_folds=10):
    """
    Réalise une validation croisée sur l'ensemble des données en utilisant un nombre de plis spécifié.
    
    Args:
        model (Pipeline): Le pipeline du modèle.
        X (pd.DataFrame): Variables explicatives.
        y (pd.Series): Variable cible.
        cv_folds (int): Nombre de plis pour la validation croisée (par défaut 10).
    
    Affiche la RMSE moyenne et son écart-type.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    neg_mse_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse_scores)
    print(f"\nValidation croisée à {cv_folds} plis :")
    print("RMSE moyen = {:.3f} (± {:.3f})".format(rmse_scores.mean(), rmse_scores.std()))

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label="Prédictions")
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Comparaison entre valeurs réelles et prédictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ligne d'identité")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = "data/processed_pipeline.csv"
    target = "Rating"
    
    df = load_data(file_path)
    print("\nAperçu des données chargées :")
    explore_data(df)
    
    X_train, X_test, y_train, y_test = split_data(file_path, target, test_size=0.2)
    print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape}")
    print(f"Taille de l'ensemble de test : {X_test.shape}")
    
    model = train_linear_regression(X_train, y_train, poly_degree=2)
    
    evaluate_model(model, X_test, y_test)
    
    cross_validate_model(model, X_train, y_train, cv_folds=10)
    
    y_pred = model.predict(X_test)
    comparison_df = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": y_pred
    })
    print("\nComparaison des valeurs réelles et prédites :")
    print(comparison_df.head(20))
    
    joblib.dump(model, 'model_pipeline.pkl')
    print("Pipeline complet enregistré dans 'model_pipeline.pkl'.")
