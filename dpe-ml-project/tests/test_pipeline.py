import pytest
import pandas as pd
from src.data.load import load_data
from src.data.preprocess import clean_data
from src.models.train import train_model
from src.models.evaluate import evaluate_model

def test_pipeline():
    """Teste l'exécution du pipeline complet sur un petit dataset mock."""
    # Créer un dataset mock
    data = pd.DataFrame({
        'Surface_habitable_logement': [100, 150, 200],
        'Qualité_isolation_enveloppe': ['bonne', 'insuffisante', 'moyenne'],
        'Etiquette_DPE': ['A', 'B', 'C']
    })
    
    # Simuler config
    config = {
        'data': {'processed_path': 'data/processed/test.csv'},
        'split': {'test_size': 0.2},
        'seed': 42,
        'model': {'params': {'n_estimators': 10}},
        'output': {'model_path': 'outputs/models/test_model.pkl'}
    }
    import yaml
    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    
    # Tester pipeline
    df = clean_data(data)
    pipeline, X_test, y_test = train_model(df)
    evaluate_model(pipeline, X_test, y_test)
    
    assert pipeline is not None
    assert len(X_test) > 0