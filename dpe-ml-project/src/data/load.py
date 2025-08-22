import pandas as pd
import yaml

def load_data(config_path='config.yaml'):
    """Charge et concatène les données train et test depuis les chemins spécifiés dans le fichier de config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train = pd.read_csv(config['data']['train_path'])
    test = pd.read_csv(config['data']['test_path'])
    df = pd.concat([train, test], ignore_index=True)
    return df
