import pandas as pd

def correlation_with_target(csv_path="data/Fusion_with_stand.csv", target="Orientation gagnant"):
    # Chargement
    df = pd.read_csv(csv_path)

    # Encodage de la cible
    if df[target].dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
    else:
        le = None

    # Colonnes à exclure explicitement (votes)
    exclude_columns = [
        'Extrême gauche (%)', 'Gauche (%)', 'Centre (%)',
        'Droite (%)', 'Extrême droite (%)', 'Orientation gagnant'
    ]

    # Sélection des colonnes numériques et corrélées
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    usable_cols = [col for col in numeric_cols if col not in exclude_columns]

    # Calcul des corrélations
    correlations = df[usable_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)

    # Affichage
    print(f"\n Corrélation des variables explicatives avec la cible '{target}':\n")
    print(correlations)

    # Variables à garder pour les modèles (forte corrélation)
    variables_utiles = correlations.index.tolist()
    return variables_utiles, le

# Pour exécution directe
if __name__ == "__main__":
    variables_utiles, _ = correlation_with_target()
