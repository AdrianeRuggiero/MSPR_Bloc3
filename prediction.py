import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from correlation_report import correlation_with_target

# === Config
DATA_PATH = "data/Fusion_with_stand.csv"
MODELS_FOLDER = "models/saved_models"

# === Chargement des données
variables_utiles, label_encoder = correlation_with_target()
df = pd.read_csv(DATA_PATH)

# Encodage cible
if df["Orientation gagnant"].dtype == object:
    le = LabelEncoder()
    df["Orientation gagnant"] = le.fit_transform(df["Orientation gagnant"])
else:
    le = label_encoder

X = df[variables_utiles]
y = df["Orientation gagnant"]

# Rhône
df["Code commune"] = df["Code commune"].astype(str)
is_rhone = df["Code commune"].str.startswith("69")
X_rhone = X[is_rhone]
X_rest = X[~is_rhone]
y_rest = y[~is_rhone]

# Standardisation
scaler = StandardScaler()
X_rest_scaled = scaler.fit_transform(X_rest)
X_rhone_scaled = scaler.transform(X_rhone)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_rest_scaled, y_rest, test_size=0.2, random_state=42)

# === Test des modèles
best_model = None
best_model_name = ""
best_accuracy = 0
scores = {}

for file in os.listdir(MODELS_FOLDER):
    if file.endswith(".pkl"):
        path = os.path.join(MODELS_FOLDER, file)
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            scores[file] = acc
            print(f"{file:<40} → Accuracy : {acc:.4f}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_model_name = file
        except Exception as e:
            print(f"Erreur avec {file} : {e}")

# === Résultats
if best_model:
    print(f"\n Meilleur modèle : {best_model_name} avec Accuracy = {best_accuracy:.4f}")

    # Prédiction Rhône
    rhone_preds = best_model.predict(X_rhone_scaled)
    rhone_labels = le.inverse_transform(rhone_preds)
    df_rhone = df[is_rhone].copy()
    df_rhone["Orientation prédite"] = rhone_labels

    counts = df_rhone["Orientation prédite"].value_counts()
    gagnant = counts.idxmax()
    print(f"\n Prédiction finale pour le Rhône : {gagnant}")

    # Camembert
    counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title=f"Prédiction Rhône - {best_model_name}")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # Barplot comparaison des modèles
    plt.figure(figsize=(10, 5))
    plt.bar(scores.keys(), scores.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Comparaison des accuracies - Tous les modèles")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
else:
    print("Aucun modèle n a pu être évalué.")
