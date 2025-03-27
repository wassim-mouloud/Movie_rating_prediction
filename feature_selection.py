import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from data_cleaning import impute_missing_values


def select_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only numeric columns since linear regression requires numerical inputs.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    print("Numerical columns retained:", numeric_df.columns.tolist())
    return numeric_df

def correlation_filter(df: pd.DataFrame, target: str, threshold: float = 0.1) -> pd.DataFrame:
    """
    Select features whose absolute correlation with the target is at least the given threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing only numeric columns.
        target (str): Name of the target column (e.g., 'Rating').
        threshold (float): Correlation threshold.
    
    Returns:
        pd.DataFrame: DataFrame reduced to features with sufficient correlation with the target.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not present in the DataFrame.")
    
    corr_matrix = df.corr()
    target_corr = corr_matrix[target].abs().sort_values(ascending=False)
    print("\nCorrelation with target:")
    print(target_corr)
    
    selected_features = target_corr[target_corr >= threshold].index.tolist()
    if target not in selected_features:
        selected_features.append(target)  # Ensure the target is included
    print(f"\nFeatures selected (threshold = {threshold}):", selected_features)
    
    return df[selected_features]

def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove features that are highly correlated with each other.
    For any pair of features with correlation above the given threshold, one is dropped.
    
    Args:
        df (pd.DataFrame): DataFrame with numeric features.
        threshold (float): Threshold for removing features with high inter-correlation.
    
    Returns:
        pd.DataFrame: DataFrame with reduced multicollinearity.
    """
    # Compute absolute correlation matrix and only consider upper triangle (excluding diagonal)
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop: if any feature has a correlation above the threshold with another feature
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    print(f"\nFeatures to drop due to high multicollinearity (threshold = {threshold}):", to_drop)
    
    return df.drop(columns=to_drop)

def feature_selection_pipeline(df: pd.DataFrame, target: str = "Rating", corr_threshold: float = 0.1, multicol_threshold: float = 0.9) -> pd.DataFrame:
    """
    Complete feature selection pipeline:
      - Retain only numeric features.
      - Filter features based on their correlation with the target.
      - Remove features with high multicollinearity.
      
    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target variable (e.g., 'Rating').
        corr_threshold (float): Minimum correlation with target to be kept.
        multicol_threshold (float): Maximum allowed correlation between features.
        
    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    df_numeric = select_numerical_features(df)
    df_filtered = correlation_filter(df_numeric, target, threshold=corr_threshold)
    df_final = remove_multicollinearity(df_filtered, threshold=multicol_threshold)
    print("Feature selection completed.")
    return df_final

def feature_selection_pipeline_lasso(df: pd.DataFrame, target: str = "Rating", cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Complete feature selection pipeline using LassoCV:
      - Retain only numeric features.
      - Apply LassoCV for feature selection.
      
    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target variable.
        cv (int): Number of folds for cross-validation.
        random_state (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    df_numeric = select_numerical_features(df)
    impute_missing_values(df_numeric, strategy="median")
    df_selected = lasso_feature_selection(df_numeric, target=target, cv=cv, random_state=random_state)
    print("Feature selection via Lasso completed.")
    return df_selected





def rfe_feature_selection(df: pd.DataFrame, target: str, n_features_to_select: int = 10) -> pd.DataFrame:
    """
    Sélectionne les variables via RFE (Recursive Feature Elimination) en utilisant une régression linéaire.
    
    Args:
        df (pd.DataFrame): DataFrame contenant uniquement des colonnes numériques.
        target (str): Nom de la variable cible.
        n_features_to_select (int): Nombre de variables à conserver.
        
    Returns:
        pd.DataFrame: DataFrame réduite aux features sélectionnées.
    """
    df = select_numerical_features(df)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.support_].tolist()
    # S'assurer que la cible est incluse
    if target not in selected_features:
        selected_features.append(target)
    
    print("\nMéthode RFE - Variables sélectionnées :", selected_features)
    return df[selected_features]



def lasso_feature_selection(df: pd.DataFrame, target: str, cv: int = 5, random_state: int = 42, max_iter: int = 10000, tol: float = 1e-4) -> pd.DataFrame:
    """
    Sélectionne les variables via LassoCV, qui effectue une régularisation L1.
    Les variables dont le coefficient est non nul sont conservées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant uniquement des colonnes numériques.
        target (str): Nom de la variable cible.
        cv (int): Nombre de plis pour la validation croisée.
        random_state (int): Graine pour la reproductibilité.
        max_iter (int): Nombre maximum d'itérations pour la convergence.
        tol (float): Tolérance pour la convergence.
        
    Returns:
        pd.DataFrame: DataFrame réduite aux features dont le coefficient n'est pas nul.
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=max_iter, tol=tol).fit(X, y)
    coef = pd.Series(lasso.coef_, index=X.columns)
    selected_features = coef[coef != 0].index.tolist()
    
    if target not in selected_features:
        selected_features.append(target)
    
    print("\nMéthode LassoCV - Variables sélectionnées :", selected_features)
    return df[selected_features]




if __name__ == "__main__":
    file_path = "data/encoded_data.csv"  # For example, the file output from your feature engineering
    try:
        df = pd.read_csv(file_path)
        print("Data loaded for feature selection.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Preview the data before selection
    print("\nData preview before selection:")
    print(df.head())
    
    # Apply the feature selection pipeline
    df_selected = feature_selection_pipeline(df, target="Rating", corr_threshold=0.1, multicol_threshold=0.9)
    
    # Preview the data after selection
    print("\nData preview after selection:")
    print(df_selected.head())
