import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Importer vos fonctions de nettoyage et feature engineering
from data_cleaning import clean_data, impute_missing_values
from feature_engineering import feature_engineering_pipeline

# Liste des variables que le modèle a vues lors de l'entraînement
SELECTED_FEATURES = [
    'No of Persons Voted', 'Duration_Min', 'Drama', 'Genre_Count',
    'Description_Word_Count', 'Description_Char_Count', 'Written_by_freq',
    'No of Persons Voted_Log', 'Release_Year', 'Film_Age', 'Release_Decade', 'Rating'
]

def process_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique manuellement les étapes de nettoyage et de feature engineering sur les données brutes,
    puis conserve uniquement les variables sélectionnées utilisées lors de l'entraînement (à l'exclusion de la cible).
    Les colonnes manquantes sont ajoutées avec la valeur 0.
    """
    # Appliquer le nettoyage
    df = clean_data(input_df)
    # Appliquer le feature engineering
    df = feature_engineering_pipeline(df)
    # Imputer les valeurs manquantes (si ce n'est pas déjà fait dans clean_data)
    df = impute_missing_values(df, strategy="median")
    
    # Définir la liste des features utilisées pour l'entraînement, sans la cible "Rating"
    features_prediction = [col for col in SELECTED_FEATURES if col != "Rating"]
    
    # Conserver uniquement les variables sélectionnées (ajoute 0 pour celles manquantes)
    df = df.reindex(columns=features_prediction, fill_value=0)
    return df


st.title("Prédicteur de Rating de Films")

st.write("Entrez les caractéristiques du film :")

# Saisies utilisateur (exemple minimal pour générer les features nécessaires)
nb_votes = st.number_input("Nombre de votes (No of Persons Voted)", min_value=0, value=1000)
duration = st.number_input("Durée (minutes)", min_value=0, value=120)
release_year = st.number_input("Année de sortie", min_value=1900, max_value=2100, value=2000)
# Pour le modèle, nous attendons également des variables générées par le pipeline
# Pour les features textuelles (Title, Description, Directed by, Written by), on peut fournir des valeurs vides
# Le pipeline se chargera de les transformer.
# Pour le genre, on peut laisser vide (ce qui donnera Genre_Count = 0 et aucun genre détecté)

# Création d'un DataFrame avec les données brutes
input_data = pd.DataFrame({
    "No of Persons Voted": [nb_votes],
    "Duration": [duration],
    "Release Date": [f"{release_year}-01-01"],
    "Rating": [0],  # Valeur factice
    "Genres": [""],
    "Title": [""],
    "Description": [""],
    "Directed by": [""],
    "Written by": [""]
})

if st.button("Prédire le rating"):
    # Appliquer le pipeline de transformation sur les données brutes
    processed_input = process_input_data(input_data)
    
    # Charger le pipeline complet (modèle) enregistré
    model = joblib.load("model_pipeline.pkl")
    
    # Effectuer la prédiction
    prediction = model.predict(processed_input)
    st.write("Le rating prédit est :", np.round(prediction[0], 2))
    
