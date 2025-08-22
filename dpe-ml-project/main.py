import src.data.load as load
import src.data.preprocess as preprocess
import src.features.build_features as features
import src.models.train as train
import src.models.evaluate as evaluate
import src.visualization.visualize as viz
import src.utils as utils

def main():
    """Exécute le pipeline complet."""
    logger = utils.setup_logging()
    logger.info("Démarrage du pipeline")
    
    # Charger les données
    df = load.load_data()
    logger.info("Données chargées")
    
    # Nettoyer les données
    df = preprocess.clean_data(df)
    logger.info("Données nettoyées")
    
    # Feature engineering
    df = features.build_features(df)
    logger.info("Features construites")
    
    # Sauvegarder les données traitées
    preprocess.save_processed_data(df)
    logger.info("Données traitées sauvegardées")
    
    # Entraîner le modèle
    pipeline, X_test, y_test, label_encoder = train.train_model(df)
    logger.info("Modèle entraîné")
    
    # Évaluer le modèle
    evaluate.evaluate_model(pipeline, X_test, y_test)
    logger.info("Modèle évalué")
    
    # Visualiser
    viz.plot_feature_importance(pipeline)
    logger.info("Visualisation générée")

if __name__ == "__main__":
    main()