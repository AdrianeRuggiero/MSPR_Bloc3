import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le fichier
df = pd.read_csv("data/Fusion_with_stand.csv")

# Colonnes socio-économiques à corréler
socio_cols = [
    'Chomage (%)',
    'Variation Moyenne Inflation',
    'Niveau Education Moyen',
    'Age dominant',
    'Entreprises',
    'Criminalité (%)'
]

# Colonnes d’orientation politique
orientation_cols = [
    'Gauche (%)',
    'Droite (%)',
    'Centre (%)',
    'Extrême gauche (%)',
    'Extrême droite (%)'
]

# Calcul des corrélations croisées
correlations = []
for vote_col in orientation_cols:
    for socio_col in socio_cols:
        corr = df[vote_col].corr(df[socio_col])
        correlations.append({
            'Orientation': vote_col.replace(" (%)", ""),
            'Variable socio-éco': socio_col,
            'Corrélation': corr
        })

# Transformation en DataFrame
corr_df = pd.DataFrame(correlations)

# === Barplot groupé
plt.figure(figsize=(10, 6))
sns.barplot(
    data=corr_df,
    x="Variable socio-éco",
    y="Corrélation",
    hue="Orientation"
)
plt.title("Corrélations entre les variables socio-économiques et les orientations politiques")
plt.axhline(0, color="black", linestyle="--")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
