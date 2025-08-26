# Projet d'évaluation de performance energétique d'un logement ou d'un bâtiment (DPE)

Le Diagnostic de Performance Energétique (DPE) renseigne sur la performance énergétique et environnementale d’un logement ou d’un bâtiment, en évaluant sa consommation d’énergie et son impact en matière d’émissions de gaz à effet de serre. Le contenu et les modalités d’établissement du DPE sont réglementés. Le DPE contient des informations sur les caractéristiques du bâtiment ou du logement (surface, orientation, murs, fenêtres, matériaux, etc.) ainsi que sur ses équipements (de chauffage, de production d’eau chaude sanitaire, de ventilation, etc.)

Pour ce challenge c'est l'étiquette DPE que l'on cherche à prédire. Cette étiquette est une valeur comprise entre A et G et permet de savoir si le bâtiment et bon ou non d'un point de vue énergétique.
Lien du challenge Kaggle : https://www.kaggle.com/competitions/esgi-x-inetum-hackathon-mars-2023/overview

# Traitement

Pour ce projet, j'ai réalisé en premier lieu un notebook où je nettoie les données brutes :
- Suppression de colonnes
- Suppression des valeurs aberrantes
- Traitement pour les valeurs manquantes :
    1. Variables catégorielles : imputation "missing" en cas de valeurs manquantes et one-hot encoding
    2. Variables numériques : imputation avec la médiane pour les valeurs manquantes et one-hot encoding

Pour l'entraînement, j'utilise le modèle XGBoost. Ensuite, j'effectue les prédictions sur un dataset fourni puis je soumets le résultat sur Kaggle, où j'ai obtenu le score de 94%. 

# Infrastructure pipeline

Après avoir travaillé sur mon modèle performance et mon modèle métier, j'ai voulu mettre en place une version plus industrialisable de ce projet. J'ai donc découpé et adapté le code Python existant pour l'exécuter via Pipeline (de Sklearn) et ColumnTransformer.
