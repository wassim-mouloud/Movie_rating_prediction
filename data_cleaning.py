import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

def drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Dropped 'Unnamed: 0' column.")
    return df

def convert_release_date(df: pd.DataFrame) -> pd.DataFrame:
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    print("Converted 'Release Date' to datetime.")
    return df

def clean_rating(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.dropna(subset=['Rating'])
    after = df.shape[0]
    print(f"Dropped {before - after} rows with missing 'Rating'.")
    return df

def convert_persons_voted(df: pd.DataFrame) -> pd.DataFrame:
    if 'No of Persons Voted' in df.columns:
        df['No of Persons Voted'] = df['No of Persons Voted'].astype(str).str.replace(r'[^\d]', '', regex=True)
        df['No of Persons Voted'] = pd.to_numeric(df['No of Persons Voted'], errors='coerce')
        print("Converted 'No of Persons Voted' to integer.")
    return df

def convert_duration(df: pd.DataFrame) -> pd.DataFrame:
    def duration_to_minutes(duration_value):
        if isinstance(duration_value, (int, float)):
            return duration_value
        if pd.isnull(duration_value):
            return np.nan
        hours = re.search(r'(\d+)\s*h', duration_value)
        minutes = re.search(r'(\d+)\s*m', duration_value)
        total_minutes = 0
        if hours:
            total_minutes += int(hours.group(1)) * 60
        if minutes:
            total_minutes += int(minutes.group(1))
        return total_minutes

    if 'Duration' in df.columns:
        df['Duration_Min'] = df['Duration'].apply(duration_to_minutes)
        print("Conversion de 'Duration' en minutes effectuée dans 'Duration_Min'.")
    return df


def remove_outliers(df: pd.DataFrame, column: str, method: str = "IQR") -> pd.DataFrame:
    if column not in df.columns:
        print(f"Column {column} not found.")
        return df
    
    if method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = df.shape[0]
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        after = df.shape[0]
        print(f"Removed {before - after} outlier rows from '{column}' using IQR.")
    else:
        print("Method not implemented.")
    return df

def scale_data(df: pd.DataFrame, columns: list, scaler_type: str = "standard") -> pd.DataFrame:
    df_scaled = df.copy()
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        print("Unrecognized scaler type; no scaling applied.")
        return df_scaled
    
    for col in columns:
        if col in df_scaled.columns:
            df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
            print(f"Scaled column '{col}' using {scaler_type} scaler.")
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return df_scaled

def impute_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Impute missing values in numeric columns using the specified strategy.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print("Imputed missing values in numeric columns using strategy:", strategy)
    return df

def impute_missing_values_multiple(df: pd.DataFrame, n_imputations: int = 5, base_random_state: int = 0) -> list:
    """
    Effectue une imputation multiple sur les colonnes numériques du DataFrame.
    Pour chaque imputation, on utilise un random_state différent afin de capturer l'incertitude.
    
    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
        n_imputations (int): Nombre de jeux de données imputés à générer (par défaut 5).
        base_random_state (int): La graine de départ pour le random_state (par défaut 0).
        
    Returns:
        list: Une liste de DataFrames avec les valeurs manquantes imputées.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    imputed_datasets = []
    
    for i in range(n_imputations):
        rs = base_random_state + i
        imputer = IterativeImputer(random_state=rs)
        imputed_array = imputer.fit_transform(df[numeric_cols])
        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputed_array
        imputed_datasets.append(df_imputed)
    
    print(f"Imputation multiple effectuée : {n_imputations} jeux de données générés.")
    return imputed_datasets


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_index_column(df)
    df = convert_release_date(df)
    df = clean_rating(df)
    df = convert_persons_voted(df)
    df = convert_duration(df)

    if 'Duration_Min' in df.columns:
        df = remove_outliers(df, 'Duration_Min')
    if 'No of Persons Voted' in df.columns:
        df = remove_outliers(df, 'No of Persons Voted')
    
    # df = scale_data(df, columns=['Rating', 'No of Persons Voted', 'Duration_Min'], scaler_type="standard")

    
    print("Global cleaning completed.")
    return df

def visualize_clean_data(df: pd.DataFrame) -> None:
    numeric_cols = ['Rating', 'No of Persons Voted', 'Duration_Min']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
        plt.subplot(1, 2, 2)
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_path = "data/raw_data.csv"  
    try:
        df = pd.read_csv(file_path)
        print("Raw data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    df_clean = clean_data(df)   
    
    print("\nPreview of cleaned data:")
    print(df_clean.head())
    print("\nData info after cleaning:")
    print(df_clean.info())
    
    print("\nVisualizing cleaned data:")
    visualize_clean_data(df_clean)
