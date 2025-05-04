import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os
from correlation_report import correlation_with_target

# === Données
variables_utiles, label_encoder = correlation_with_target()
DATA_PATH = "data/Fusion_with_stand.csv"
MODEL_PATH = "models/saved_models/best_model_xgboost.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)

if df["Orientation gagnant"].dtype == object:
    le = LabelEncoder()
    df["Orientation gagnant"] = le.fit_transform(df["Orientation gagnant"])
else:
    le = label_encoder

X = df[variables_utiles]
y = df["Orientation gagnant"]

df["Code commune"] = df["Code commune"].astype(str)
is_rhone = df["Code commune"].str.startswith("69")
X_rhone = X[is_rhone]
X_rest = X[~is_rhone]
y_rest = y[~is_rhone]

scaler = StandardScaler()
X_rest_scaled = scaler.fit_transform(X_rest)
X_rhone_scaled = scaler.transform(X_rhone)

X_train, X_val, y_train, y_val = train_test_split(X_rest_scaled, y_rest, test_size=0.2, random_state=42)

# === Entraînement simple avec test de 10 profondeurs
best_accuracy = 0
best_model = None
accuracies = []

for epoch, depth in enumerate(range(3, 13), 1):
    model = XGBClassifier(
        max_depth=depth,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Époque {epoch}/10 - max_depth={depth} - Accuracy : {acc:.4f}")
    accuracies.append(acc)
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# === Sauvegarde
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f"\n Modèle XGBoost sauvegardé dans {MODEL_PATH} (accuracy = {best_accuracy:.4f})")

# === Graphique accuracy
plt.plot(range(3, 13), accuracies, marker='o')
plt.title("Accuracy selon max_depth - XGBoost")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Évaluation finale
y_val_pred = best_model.predict(X_val)
print("\nMatrice de confusion :")
sns.heatmap(confusion_matrix(y_val, y_val_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion - XGBoost")
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

counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="Prédiction Rhône - XGBoost")
plt.ylabel("")
plt.tight_layout()
plt.show()
