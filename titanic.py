
# preparation des données
# Importation des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Chargement du dataset Titanic
df = pd.read_csv('Titanic-Dataset.csv')

print(df.head())
print(df.info())

#Quelle est la repartition des passagers par class (Pclass)
print(df['Pclass'].value_counts())

# Quel est le pourcentage de survivants
print(df['Survived'].mean())

# donne le nb de survivant homme
survivants_hommes = df[(df['Sex'] == 'male') & (df['Survived'] == 1)]

print(df['Sex'].value_counts())

print(df['Fare'].mean())

print(df.loc[df['Age'].idxmax(), 'Name'])

print(df['Embarked'].value_counts())

# Exploration initiale et nettoyage des données
# Remplacement des valeurs manquantes dans 'Age' par la médiane
df['Age'] = df['Age'].fillna(df['Age'].median())

# Remplacement des valeurs manquantes dans 'Embarked' par la valeur la plus fréquente (modale)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Suppression de la colonne 'Cabin' car elle contient trop de valeurs manquantes
df.drop(columns=['Cabin'], inplace=True)

# Encodage des variables catégorielles (Sex et Embarked)
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Conversion du sexe en valeurs numériques
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # Conversion des ports d'embarquement

# Définition des variables explicatives (X) et de la variable cible (y)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]  # Variables explicatives
y = df['Survived']  # Variable cible (survie)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement du modèle d'arbre de décision
decision_tree = DecisionTreeClassifier(random_state=30)  # Initialisation avec un état aléatoire pour des résultats reproductibles
decision_tree.fit(X_train, y_train)  # Entraînement du modèle sur l'ensemble d'entraînement

# Prédictions sur l'ensemble de test
y_pred = decision_tree.predict(X_test)  # Prédiction de la survie des passagers dans l'ensemble de test

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)  # Calcul de la précision du modèle
classification_rep = classification_report(y_test, y_pred)  # Rapport détaillé sur les performances
conf_matrix = confusion_matrix(y_test, y_pred)  # Matrice de confusion pour évaluer les prédictions correctes et incorrectes

# Affichage des résultats
print("Précision du modèle :", accuracy)
print("\nRapport de classification :\n", classification_rep)
print("\nMatrice de confusion :\n", conf_matrix)

# Importation de la fonction pour visualiser l'arbre
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Création et entraînement du modèle d'arbre de décision avec paramètres pour simplifier l'arbre
decision_tree = DecisionTreeClassifier(
    max_depth=3,               # Limite la profondeur à 3 niveaux
    min_samples_split=10,      # Au moins 10 échantillons requis pour diviser un nœud
    min_samples_leaf=5,        # Au moins 5 échantillons par feuille
    random_state=42
)
decision_tree.fit(X_train, y_train)

# Visualisation de l'arbre réduit
plt.figure(figsize=(12, 8))
plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()


