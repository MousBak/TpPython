# TpPython

Projet de Prédiction de Survie des Passagers du Titanic
Ce projet utilise un modèle d'arbre de décision pour prédire la survie des passagers du Titanic en fonction de différentes caractéristiques comme la classe, le sexe, l'âge, le nombre de membres de la famille à bord, le tarif payé et le port d'embarquement. Ce projet permet d'explorer les données, de préparer et de nettoyer les données, de construire un modèle de machine learning, et de visualiser les résultats.

Structure du Projet
Chargement et Exploration des Données : Chargement des données Titanic et première exploration des caractéristiques du dataset.
Préparation et Nettoyage des Données :
Remplissage des valeurs manquantes dans les colonnes Age et Embarked.
Suppression de la colonne Cabin en raison d'un grand nombre de valeurs manquantes.
Encodage des variables catégorielles (Sex et Embarked) en valeurs numériques.

Modélisation :
Division des données en ensembles d'entraînement et de test.
Entraînement d'un modèle d'arbre de décision pour prédire la survie.
Ajustement de l'arbre pour réduire sa taille et améliorer la lisibilité.
Évaluation :
Précision, rapport de classification, et matrice de confusion.
Visualisation de l'arbre de décision.

Datasets
Les données proviennent du célèbre dataset Titanic, disponible sur Kaggle.

Colonnes du Dataset
PassengerId : Identifiant unique du passager
Survived : Indicateur de survie (1 = Survécu, 0 = Non survécu)
Pclass : Classe de la cabine du passager (1ère, 2ème, ou 3ème classe)
Name : Nom du passager
Sex : Sexe du passager
Age : Âge du passager
SibSp : Nombre de frères/soeurs ou conjoint(e) à bord
Parch : Nombre de parents/enfants à bord
Ticket : Numéro du billet
Fare : Tarif payé pour le billet
Cabin : Numéro de cabine (souvent manquant)
Embarked : Port d'embarquement (C = Cherbourg, Q = Queenstown, S = Southampton)
Installation
Clonez le dépôt :

bash
Copier le code
Installez les dépendances requises :

bash
Copier le code
pip install pandas scikit-learn matplotlib seaborn
Utilisation
Préparation des Données : Le code prépare le dataset en remplissant les valeurs manquantes et en encodant les variables catégorielles.
Entraînement du Modèle : Le modèle d'arbre de décision est entraîné sur les données nettoyées.
Visualisation de l'Arbre de Décision : L'arbre est visualisé pour interpréter les règles de décision.
Évaluation : Le modèle est évalué en calculant la précision et en affichant la matrice de confusion.
Pour exécuter le projet, lancez le script principal dans votre environnement Python :

bash
Copier le code
python titanic_decision_tree.py
Résultats et Visualisation
Les résultats montrent la précision du modèle, le rapport de classification et la matrice de confusion. L'arbre de décision est également visualisé pour comprendre les décisions du modèle en fonction des différentes caractéristiques des passagers.

Exemples de Visualisation
Voici un aperçu de certaines visualisations disponibles dans ce projet :

Distribution de l'âge et du tarif payé.
Répartition des survivants selon la classe et le sexe.
Arbre de décision réduit pour faciliter la lecture.
Améliorations Futures
Optimisation des hyperparamètres pour améliorer la précision.
Exploration d'autres modèles de machine learning (forêts aléatoires, régressions logistiques).
Intégration d'une interface pour prédire la survie en fonction des données saisies par l'utilisateur.
