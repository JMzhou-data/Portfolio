import pandas as pd 
import joblib
from sklearn.metrics import accuracy_score, classification_report
import os

val_path = "../data/raw/val.csv"
model_path = "../outputs/models/xgboost_model.pkl"
encoder_path = "../outputs/models/label_encoder.pkl"
submission_path = "../outputs/prediction/submission.csv"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modèle non trouvé à {model_path}")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(f"Label encoder non trouvé à {encoder_path}")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Fichier val.csv non trouvé à {val_path}")

df_val = pd.read_csv(val_path)
print(f"Colonnes disponibles : {df_val.columns.tolist()}")
print(f"Nombre de lignes : {len(df_val)}")

model = joblib.load(model_path)
print("Modèle chargé avec succès")

label_encoder = joblib.load(encoder_path)
print("Label encoder chargé avec succès. Classes : ", label_encoder.classes_)

# Prédiction 
y_pred_encoded = model.predict(df_val)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
print("Exemple de prédictions : ", y_pred[:5])

# Création du DataFrame de soumission
submission = pd.DataFrame({
    'N°DPE': df_val['N°DPE'] if 'N°DPE' in df_val.columns else df_val.index,
    'Etiquette_DPE': y_pred
})

# Sauvegarde
submission.to_csv(submission_path, index=False)
print(f"Fichier de soumission généré : {submission_path}")
print(submission.head())