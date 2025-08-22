import yaml
import logging
import os

def load_config(config_path='config.yaml'):
    """Charge le fichier de configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    """Configure le logging."""
    # On vérifie l'existence du répertoire outputs/
    os.makedirs('outputs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/project.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)