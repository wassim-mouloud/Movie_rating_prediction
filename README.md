# 🎬 Prédicteur de Ratings de Films

Bienvenue dans ce projet de prédiction des ratings de films ! Ce projet utilise un pipeline complet de traitement des données et de modélisation pour prédire les notes des films à partir de leurs métadonnées.

## 🔮 Fonctionnement du Pipeline

Le pipeline est organisé en plusieurs étapes clés :

- **Nettoyage des données :** Transformation des types, suppression des outliers et imputation des valeurs manquantes.
- **Feature Engineering :** Encodage des variables textuelles (TF-IDF, extraction de métriques) et catégorielles, ainsi que l'extraction d'informations temporelles et de statistiques supplémentaires (ex. : nombre de genres, durée, votes transformés).
- **Sélection des Variables :** Filtrage des features en fonction de leur corrélation avec le rating et réduction de la multicolinéarité.
- **Modélisation :** Entraînement d’un modèle de régression linéaire enrichi par des transformations polynomiales.
- **Validation & Enregistrement :** Validation croisée pour évaluer le modèle et enregistrement du pipeline complet pour une utilisation future.

---

## 💻 Interface Graphique avec Streamlit (Linear Regression model)

Pour faciliter l'utilisation de notre prédicteur, nous avons développé une interface graphique interactive avec **Streamlit**. Voici comment l'utiliser :

1. **Installation :**  
   Installez les dépendances via le fichier `requirements.txt` en utilisant la commande :

   ```
   pip install -r requirements.txt
   ```

2. **Lancement de l'Interface :**  
   Dans votre terminal, lancez l'application avec :

   ```
   streamlit run app.py
   ```

   Votre navigateur s’ouvrira automatiquement et vous pourrez saisir les caractéristiques d’un film.

3. **Utilisation :**
   - **Saisissez les informations** demandées (nombre de votes, durée, année de sortie, etc.).
   - **Cliquez sur le bouton** "Prédire le rating" pour obtenir la prédiction.
   - Le pipeline complet (nettoyage, feature engineering, sélection, modélisation) sera appliqué automatiquement pour générer le rating prédit.
