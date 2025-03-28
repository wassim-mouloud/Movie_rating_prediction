import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_cleaning import clean_data, impute_missing_values
from feature_engineering import feature_engineering_pipeline

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
    df = clean_data(input_df)
    df = feature_engineering_pipeline(df)
    df = impute_missing_values(df, strategy="median")
    
    features_prediction = [col for col in SELECTED_FEATURES if col != "Rating"]
    
    df = df.reindex(columns=features_prediction, fill_value=0)
    return df


st.title("Prédicteur de Rating de Films")

st.write("Entrez les caractéristiques du film :")

nb_votes = st.number_input("Nombre de votes (No of Persons Voted)", min_value=0, value=1000)
duration = st.number_input("Durée (minutes)", min_value=0, value=120)
release_year = st.number_input("Année de sortie", min_value=1900, max_value=2100, value=2000)

input_data = pd.DataFrame({
    "No of Persons Voted": [nb_votes],
    "Duration": [duration],
    "Release Date": [f"{release_year}-01-01"],
    "Rating": [0],  
    "Genres": [""],
    "Title": [""],
    "Description": [""],
    "Directed by": [""],
    "Written by": [""]
})

if st.button("Prédire le rating"):
    processed_input = process_input_data(input_data)
    
    model = joblib.load("model_pipeline.pkl")
    
    prediction = model.predict(processed_input)
    st.write("Le rating prédit est :", np.round(prediction[0], 2))
    
