
#  MSPR Bloc 3 – Prédiction d’orientation politique par commune

Ce projet vise à prédire l’**orientation politique gagnante** (Centre, Gauche, Droite, etc.) dans chaque commune française à partir de **données socio-économiques**.  
Il repose sur l’analyse de corrélations et l’entraînement de plusieurs modèles de machine learning.

---

##  Structure du projet

```
MSPR_Bloc3/
│
├── data/                      → Données d’entrée (non suivies sur GitHub)
│   └── Fusion_with_stand.csv
│
├── models/                   → Modèles prédictifs sauvegardés et scripts d'entraînement
│   ├── knn_model.py
│   ├── logistic_model.py
│   ├── mlp_model.py
│   ├── random_forest_model.py
│   ├── gradientboost_model.py
│   ├── xgboost_model.py
│   └── saved_models/          → Pickle des meilleurs modèles
│
├── correlation_report.py     → Sélection automatique des variables corrélées à la cible
├── predictions.py            → Compare tous les modèles et affiche le meilleur pour le Rhône
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Objectif

> Utiliser des indicateurs comme le chômage, le niveau d'éducation, l'inflation ou la démographie pour **prédire le parti politique gagnant** dans un departement français.

---

##  Données utilisées

- **Source principale :** fichier `Fusion_with_stand.csv`
- Colonnes incluses :
  - `Chômage (%)`, `Criminalité (%)`
  - `Variation Moyenne Inflation`
  - `Niveau d’éducation moyen`
  - `Entreprises`, `Age dominant`
  - `Orientation gagnant` (cible)

---

##  Modèles utilisés

Chacun des modèles est entraîné avec les **meilleures variables** (corrélées à la cible) et évalué sur un jeu de validation :

-  Random Forest
-  Logistic Regression (variation du paramètre `C`)
-  K-Nearest Neighbors (variation de `k`)
-  Multi-layer Perceptron (réseau de neurones simple)
-  Gradient Boosting (variation de `learning_rate`)
-  XGBoost (variation de `max_depth`)

> Chaque modèle est sauvegardé dans `models/saved_models/` au format `.pkl`.

---

##  Comparaison automatique

Le fichier `predictions.py` permet de :

- Charger tous les modèles entraînés
- Comparer leur **accuracy sur validation**
- Afficher un **camembert des prédictions pour la région Rhône**
- Générer un **graphique comparatif des performances**

---

##  Exemple d'exécution

```bash
python predictions.py
```

 `best_model_xgboost.pkl` sélectionné  
 Accuracy : `0.6523`  
 Prédiction Rhône : `Extrême droite`

---

##  Remarques

- Le fichier `.csv` n’est **pas versionné** (trop volumineux / données sensibles).
- Le dossier `data/` doit être rempli manuellement.
- Le code utilise `StandardScaler`, `LabelEncoder` et `train_test_split` pour une préparation cohérente des données.
