import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Fonctions d'encodage et transformations de base
# ---------------------------

def encode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode la colonne 'Genres' en colonnes binaires et ajoute 'Genre_Count'.
    """
    if 'Genres' not in df.columns:
        print("La colonne 'Genres' n'existe pas.")
        return df

    df['Genres_list'] = df['Genres'].apply(lambda x: [genre.strip() for genre in x.split(',')] if pd.notnull(x) else [])
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(df['Genres_list']),
                                  columns=mlb.classes_,
                                  index=df.index)
    df = pd.concat([df, genres_encoded], axis=1)
    df['Genre_Count'] = df['Genres_list'].apply(len)
    df = df.drop(columns=['Genres', 'Genres_list'])
    print("Encodage de 'Genres' et création de 'Genre_Count' effectués.")
    return df

def encode_title(df: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
    """
    Encode la colonne 'Title' avec TF-IDF et ajoute 'Title_Word_Count' et 'Title_Char_Count'.
    """
    titles = df['Title'].fillna("")
    vectorizer = TfidfVectorizer(max_features=max_features)
    title_tfidf = vectorizer.fit_transform(titles)
    title_df = pd.DataFrame(title_tfidf.toarray(),
                            columns=[f"Title_{word}" for word in vectorizer.get_feature_names_out()],
                            index=df.index)
    df = pd.concat([df, title_df], axis=1)
    df['Title_Word_Count'] = titles.apply(lambda x: len(x.split()))
    df['Title_Char_Count'] = titles.apply(lambda x: len(x))
    print("Encodage de 'Title' avec TF-IDF et ajout de 'Title_Word_Count' et 'Title_Char_Count' effectués.")
    df = df.drop(columns=['Title'])
    return df

def encode_description(df: pd.DataFrame, max_features: int = 200) -> pd.DataFrame:
    """
    Encode la colonne 'Description' avec TF-IDF et ajoute 'Description_Word_Count' et 'Description_Char_Count'.
    """
    descriptions = df['Description'].fillna("")
    vectorizer = TfidfVectorizer(max_features=max_features)
    desc_tfidf = vectorizer.fit_transform(descriptions)
    desc_df = pd.DataFrame(desc_tfidf.toarray(),
                           columns=[f"Desc_{word}" for word in vectorizer.get_feature_names_out()],
                           index=df.index)
    df = pd.concat([df, desc_df], axis=1)
    df['Description_Word_Count'] = descriptions.apply(lambda x: len(x.split()))
    df['Description_Char_Count'] = descriptions.apply(lambda x: len(x))
    print("Encodage de 'Description' avec TF-IDF et ajout de 'Description_Word_Count' et 'Description_Char_Count' effectués.")
    df = df.drop(columns=['Description'])
    return df

def encode_directed_by(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique un encodage par fréquence sur la colonne 'Directed by'.
    """
    if 'Directed by' not in df.columns:
        print("La colonne 'Directed by' n'existe pas.")
        return df
    freq = df['Directed by'].value_counts().to_dict()
    df['Directed_by_freq'] = df['Directed by'].map(freq)
    df = df.drop(columns=['Directed by'])
    print("Encodage par fréquence de 'Directed by' effectué.")
    return df

def encode_written_by(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique un encodage par fréquence sur la colonne 'Written by'.
    """
    if 'Written by' not in df.columns:
        print("La colonne 'Written by' n'existe pas.")
        return df
    freq = df['Written by'].value_counts().to_dict()
    df['Written_by_freq'] = df['Written by'].map(freq)
    df = df.drop(columns=['Written by'])
    print("Encodage par fréquence de 'Written by' effectué.")
    return df

def transform_votes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique une transformation logarithmique à la colonne 'No of Persons Voted' pour réduire l'asymétrie.
    """
    if 'No of Persons Voted' in df.columns:
        df['No of Persons Voted_Log'] = df['No of Persons Voted'].apply(lambda x: np.log(x + 1))
        print("Transformation log de 'No of Persons Voted' appliquée (nouvelle colonne : 'No of Persons Voted_Log').")
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'encodage aux colonnes catégorielles pertinentes.
    """
    df = encode_genres(df)
    df = encode_title(df, max_features=100)
    df = encode_description(df, max_features=200)
    df = encode_directed_by(df)
    df = encode_written_by(df)
    df = transform_votes(df)
    return df

# ---------------------------
# Transformations complémentaires
# ---------------------------

def extract_release_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait l'année de 'Release Date' et stocke le résultat dans 'Release_Year'.
    """
    if 'Release Date' in df.columns:
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        df['Release_Year'] = df['Release Date'].dt.year
        print("Extraction de 'Release_Year' effectuée.")
    else:
        print("Colonne 'Release Date' non trouvée.")
    return df

def add_film_age(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Ajoute une nouvelle feature 'Film_Age' calculée comme current_year - Release_Year.
    """
    if 'Release_Year' in df.columns:
        df['Film_Age'] = current_year - df['Release_Year']
        print("Feature 'Film_Age' ajoutée.")
    else:
        print("Colonne 'Release_Year' non disponible pour calculer 'Film_Age'.")
    return df

def categorize_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée 'Duration_Category' à partir de 'Duration_Min'.
    """
    if 'Duration_Min' in df.columns:
        def duration_category(minutes):
            if pd.isnull(minutes):
                return None
            if minutes < 90:
                return 'Court'
            elif minutes <= 150:
                return 'Moyen'
            else:
                return 'Long'
        df['Duration_Category'] = df['Duration_Min'].apply(duration_category)
        print("Création de 'Duration_Category' effectuée.")
    else:
        print("Colonne 'Duration_Min' non trouvée.")
    return df

def extract_release_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait le jour de la semaine de 'Release Date' et le stocke dans 'Release_DayOfWeek' (0=lundi, 6=dimanche).
    """
    if 'Release Date' in df.columns:
        df['Release_DayOfWeek'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.dayofweek
        print("Extraction du jour de la semaine ('Release_DayOfWeek') effectuée.")
    else:
        print("Colonne 'Release Date' non trouvée.")
    return df

def extract_release_decade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait la décennie à partir de 'Release_Year' et stocke le résultat dans 'Release_Decade'.
    """
    if 'Release_Year' in df.columns:
        df['Release_Decade'] = (df['Release_Year'] // 10) * 10
        print("Extraction de 'Release_Decade' effectuée.")
    else:
        print("Colonne 'Release_Year' non trouvée.")
    return df

def count_directors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compte le nombre de réalisateurs dans 'Directed by' et ajoute 'Director_Count'.
    """
    if 'Directed by' in df.columns:
        df['Director_Count'] = df['Directed by'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
        print("Calcul de 'Director_Count' effectué.")
    else:
        print("Colonne 'Directed by' non trouvée.")
    return df

def count_writers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compte le nombre de scénaristes dans 'Written by' et ajoute 'Writer_Count'.
    """
    if 'Written by' in df.columns:
        df['Writer_Count'] = df['Written by'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
        print("Calcul de 'Writer_Count' effectué.")
    else:
        print("Colonne 'Written by' non trouvée.")
    return df

def compute_avg_word_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la longueur moyenne des mots dans la colonne 'Description' et ajoute 'Description_AvgWordLength'.
    """
    if 'Description' in df.columns:
        def avg_word_length(text):
            words = text.split()
            return np.mean([len(word) for word in words]) if words else 0
        df['Description_AvgWordLength'] = df['Description'].fillna("").apply(avg_word_length)
        print("Calcul de 'Description_AvgWordLength' effectué.")
    else:
        print("Colonne 'Description' non trouvée.")
    return df

# def extract_title_sentiment(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calcule le score de sentiment de 'Title' à l'aide de TextBlob et ajoute 'Title_Sentiment'.
#     (Note: TextBlob est principalement conçu pour l'anglais.)
#     """
#     try:
#         from textblob import TextBlob
#     except ImportError:
#         print("TextBlob n'est pas installé. Installez-le pour utiliser l'analyse de sentiment.")
#         return df

#     if 'Title' in df.columns:
#         df['Title_Sentiment'] = df['Title'].fillna("").apply(lambda x: TextBlob(x).sentiment.polarity)
#         print("Extraction du sentiment du 'Title' effectuée.")
#     else:
#         print("Colonne 'Title' non trouvée.")
#     return df

# ---------------------------
# Pipeline complet de feature engineering
# ---------------------------

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering :
      - Encodage des variables catégorielles (et extraction de métriques textuelles)
      - Extraction de l'année de sortie
      - Ajout de l'âge du film
      - Catégorisation de la durée
      - Extraction du jour de la semaine
      - Extraction de la décennie de sortie
      - Comptage du nombre de réalisateurs et scénaristes
      - Calcul de la longueur moyenne des mots dans la description
      - Extraction du sentiment du titre
    """
    # Encodage et transformations de base
    df = encode_categoricals(df)
    # Transformations complémentaires
    df = extract_release_year(df)
    df = add_film_age(df)
    df = categorize_duration(df)
    # Nouvelles extractions
    df = extract_release_day_of_week(df)
    df = extract_release_decade(df)
    df = count_directors(df)
    df = count_writers(df)
    df = compute_avg_word_length(df)
    # df = extract_title_sentiment(df)
    print("Feature engineering pipeline complet terminé.")
    return df

# ---------------------------
# Bloc principal pour test (optionnel)
# ---------------------------

if __name__ == "__main__":
    file_path = "data/clean_data.csv"
    try:
        df = pd.read_csv(file_path)
        print("Données chargées pour le feature engineering.")
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        exit(1)
    
    print("\nAperçu avant feature engineering:")
    print(df.head())
    
    df_fe = feature_engineering_pipeline(df)
    
    print("\nAperçu après feature engineering:")
    print(df_fe.head())
    
    # Sauvegarder le résultat pour la modélisation
    output_file = "data/encoded_data.csv"
    df_fe.to_csv(output_file, index=False)
    print(f"\nLes données transformées ont été sauvegardées dans : {output_file}")
