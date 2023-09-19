# Projet 7 Openclassrooms : Implémentez un modèle de scoring 
#### Librairies Data Science, Analyse Exploratoire des donneés, Scoring, API REST, FLASK, STREAMLIT
## Contexte
L’entreprise Prêt à dépenser souhaite développer un modèle de scoring de la probabilité de défaut de paiement du client pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

L'application répond au cahier des charges suivant :

- Permettre de visualiser le score et l’interprétation de ce score pour chaque client pour une personne non experte en data science.
- Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
- Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

## Compétences 
- Déployer un modèle via une API dans le Web
- Utiliser un logiciel de version de code pour assurer l’intégration du modèle
- Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
- Réaliser un dashboard pour présenter son travail de modélisation
-  Présenter son travail de modélisation à l'oral

## Les données 
Les données sont disponibles sur Kaggle : https://www.kaggle.com/c/home-credit-default-risk/data

Pour ce projet, les données ont été manipulées en Python sur support Jupyter Notebook à partir d'un notebook de départ disponible également sur Kaggle, avec développement de l'application sur Flask (pour l'API) et Streamlit (pour le Dashboard). 

Le déploiement s'est effectué via Heroku. 

## Les fichiers requis pour le déploiement de l'application sont les suivants : 
1. setup.sh
2. Procfile
3. requirements.txt

## Livrables 
- Dossier Notebook Files : contient les notebooks d'analyse exploratoire (EDA), de modélisation , d'étude du data drift
- Dossier app_deploy : contient les fichiers nécessaires au déploiement sur le cloud de l'API (via Flask) et de l'application du dashboard (via streamlit)
- Dossier app_local : contient les fichiers nécessaires au déploiment de l'API et du dashboard en local
- Note méthodologique expliquant le processus et les résultats obtenus : Dabidin_Keshika_3_Note_méthodologique_082023.pdf
- Un support de présentation : Dabidin_Keshika_4_presentation_082023.pdf


