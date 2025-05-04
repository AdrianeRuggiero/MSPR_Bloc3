import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 👉 permet d'importer depuis la racine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
from correlation_report import correlation_with_target

# === Chargement des variables corrélées
variables_utiles, label_encoder = correlation_with_target()

# === Chemins
DATA_PATH = "data/Fusion_with_stand.csv"
MODEL_PATH = "models/saved_models/best_model_random_forest.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# === Chargement des données
df = pd.read_csv(DATA_PATH)

# Encodage de la cible
if df["Orientation gagnant"].dtype == object:
    le = LabelEncoder()
    df["Orientation gagnant"] = le.fit_transform(df["Orientation gagnant"])
else:
    le = label_encoder

X = df[variables_utiles]
y = df["Orientation gagnant"]

# === Séparation Rhône
df["Code commune"] = df["Code commune"].astype(str)
is_rhone = df["Code commune"].str.startswith("69")
X_rhone = X[is_rhone]
X_rest = X[~is_rhone]
y_rest = y[~is_rhone]

# === Standardisation
scaler = StandardScaler()
X_rest_scaled = scaler.fit_transform(X_rest)
X_rhone_scaled = scaler.transform(X_rhone)

# === Split
X_train, X_val, y_train, y_val = train_test_split(X_rest_scaled, y_rest, test_size=0.2, random_state=42)

# === Entraînement
best_accuracy = 0
best_model = None
accuracies = []

for epoch in range(1, 11):
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42 + epoch)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = (y_pred == y_val).mean()
    print(f"Époque {epoch}/10 - Accuracy : {acc:.4f}")
    accuracies.append(acc)
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# === Sauvegarde
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f" Modèle Random Forest sauvegardé dans {MODEL_PATH}")

# === Graphe accuracy
plt.figure()
plt.plot(range(1, 11), accuracies, marker='o')
plt.title("Évolution de l'Accuracy - Random Forest")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Matrice de confusion
y_val_pred = best_model.predict(X_val)
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matrice de confusion - Random Forest")
plt.tight_layout()
plt.show()

print("\nRapport de classification :")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# === Prédiction Rhône + camembert
rhone_preds = best_model.predict(X_rhone_scaled)
rhone_labels = le.inverse_transform(rhone_preds)
df_rhone = df[is_rhone].copy()
df_rhone["Orientation prédite"] = rhone_labels

counts = df_rhone["Orientation prédite"].value_counts()
gagnant = counts.idxmax()
print(f"\n Prédiction finale pour le Rhône : {gagnant}")

# Camembert
counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="Prédiction Rhône - Random Forest")
plt.ylabel("")
plt.tight_layout()
plt.show()
