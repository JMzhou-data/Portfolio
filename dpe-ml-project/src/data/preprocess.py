import pandas as pd
import yaml 

def clean_data(df):
    """Nettoie les données : supprime colonnes inutiles, gère valeurs aberrantes si nécessaire."""
    # Supprimer colonnes inutiles
    columns_to_drop = ["Unnamed: 0", "Facteur_couverture_solaire_saisi","Code_postal_(BAN)", "Code_postal_(brut)",
                       "N°DPE","Code_INSEE_(BAN)", "Qualité_isolation_plancher_haut_comble_aménagé","Nom__commune_(Brut)"] 
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Gérer valeurs aberrantes (exemple simple, à affiner selon ton dataset)
    df = df[df['Surface_habitable_logement'] > 0] if 'Surface_habitable_logement' in df.columns else df
    df = df[df["Surface_habitable_logement"]<=10000]
    df = df[df['Conso_5_usages/m²_é_finale']<=15000]
    df = df[df['Hauteur_sous-plafond'] < 10]
    df = df[df['Emission_GES_éclairage'] <= 1500]
    
    return df

def save_processed_data(df, config_path='config.yaml'):
    """Sauvegarde les données prétraitées."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    df.to_csv(config['data']['processed_path'], index=False)
    return df