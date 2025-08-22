import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import pandas as pd
import joblib
import os

def plot_feature_importance(pipeline, config_path='config.yaml'):
    """Trace l'importance des caractéristiques.
    
    Args:
        pipeline: Pipeline entraîné.
        config_path: Chemin vers le fichier de configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Récupérer les feature importances et noms des features
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Trier par importance
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.title("Feature Importances")
    bars = plt.barh(range(len(importances)), importances[indices], color="r", align="center")
    
    for bar in bars:
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}',
            va='center',
            ha='left'
        )
    
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.ylabel('Features')
    
    # Assurer que le dossier de sortie existe
    os.makedirs(os.path.dirname(config['output']['figure_path']), exist_ok=True)
    plt.savefig(config['output']['figure_path'], bbox_inches='tight')
    plt.close()