import pandas as pd

def build_features(df):
    """Ajoute des features dérivées si nécessaire """
    if 'Surface_habitable_immeuble' in df.columns and 'Surface_habitable_logement' in df.columns:
        df['ratio_surface_logement_immeuble'] = df['Surface_habitable_logement'] / df['Surface_habitable_immeuble'].replace(0, 1)
    
    return df