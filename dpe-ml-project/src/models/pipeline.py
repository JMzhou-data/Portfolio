from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

def get_pipeline(params, df):
    """Crée un pipeline avec prétraitement et modèle XGBoost."""
    # Définir les colonnes numériques et catégorielles (à adapter selon ton dataset)
    numeric_features = [
        'Surface_habitable_desservie_par_installation_ECS',
       'Emission_GES_éclairage', 'Conso_5_usages_é_finale_énergie_n°2',
       'Surface_totale_capteurs_photovoltaïque',
       'Conso_chauffage_dépensier_installation_chauffage_n°1',
       'Coût_chauffage_énergie_n°2', 'Emission_GES_chauffage_énergie_n°2',
       'Facteur_couverture_solaire', 'Année_construction',
       'Conso_5_usages/m²_é_finale', 'Conso_5_usages_é_finale',
       'Hauteur_sous-plafond', 'Surface_habitable_immeuble',
       'Surface_habitable_logement'
    ]
    categorical_features = [
        'Configuration_installation_chauffage_n°2', 'Cage_d\'escalier',
       'Type_générateur_froid', 'Type_émetteur_installation_chauffage_n°2',
       'Type_énergie_n°3', 'Etiquette_GES',
       'Type_générateur_n°1_installation_n°2',
       'Description_générateur_chauffage_n°2_installation_n°2',
       'Classe_altitude', 'N°_département_(BAN)',
       'Qualité_isolation_enveloppe', 'Qualité_isolation_menuiseries',
       'Qualité_isolation_murs', 'Qualité_isolation_plancher_bas',
       'Qualité_isolation_plancher_haut_comble_perdu',
       'Qualité_isolation_plancher_haut_toit_terrase', 'Type_bâtiment'
    ]

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [col for col in numeric_features if col in df.columns]),
            ('cat', categorical_transformer, [col for col in categorical_features if col in df.columns])
        ])

    # Pipeline complet
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**params))
    ])
    return pipeline