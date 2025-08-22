from sklearn.metrics import accuracy_score, classification_report
import joblib
import yaml
import os

def evaluate_model(pipeline, X_test, y_test, config_path='config.yaml'):
    """Évalue le modèle et sauvegarde les résultats.
    
    Args:
        pipeline: Pipeline entraîné.
        X_test: Données de test (features).
        y_test: Données de test (cible encodée).
        config_path: Chemin vers le fichier de configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Faire des prédictions
    y_pred = pipeline.predict(X_test)
    
    # Charger l'encodeur pour décoder les étiquettes
    label_encoder_path = 'outputs/models/label_encoder.pkl'
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
    else:
        raise FileNotFoundError(f"Label encoder non trouvé à : {label_encoder_path}")
    
    # Afficher les métriques
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test_decoded, y_pred_decoded))
    
    # Sauvegarde du modèle
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(pipeline, config['output']['model_path'])