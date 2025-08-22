from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .pipeline import get_pipeline
import yaml
import pandas as pd
import joblib
import os

def train_model(df: pd.DataFrame, config_path: str = 'config.yaml') -> tuple:
    """Entraîne le modèle sur les données fournies.
    
    Args:
        df: DataFrame contenant les données.
        config_path: Chemin vers le fichier de configuration.
    
    Returns:
        tuple: (pipeline, X_test, y_test, label_encoder)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Vérifier que la colonne cible existe
    target_column = 'Etiquette_DPE'
    if target_column not in df.columns:
        raise ValueError(f"Colonne cible '{target_column}' non trouvée dans le DataFrame")
    
    # Séparer features et cible
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column]
    
    # Encoder la variable cible
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Sauvegarder l'encodeur pour utilisation ultérieure
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(label_encoder, 'outputs/models/label_encoder.pkl')
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=config['split']['test_size'], random_state=config['seed']
    )
    
    # Créer et entraîner le pipeline
    pipeline = get_pipeline(config['model']['params'], df)
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test, label_encoder