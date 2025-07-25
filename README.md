
# Projet de Prédiction du Prix du Bitcoin

Ce projet vise à prédire le prix de clôture quotidien du Bitcoin en utilisant des données historiques. Nous allons explorer les données, prétraiter les caractéristiques (y compris les caractéristiques décalées), entraîner un modèle de régression et évaluer ses performances.

## 1. Structure du Projet

Le projet est organisé comme suit :

```
/predictionbitcoin
|-- /data
|   |-- /raw
|   |   `-- BTC-USD.csv       # Données brutes du prix du Bitcoin
|   `-- /processed
|       |-- X_train.csv       # Caractéristiques d'entraînement prétraitées
|       |-- X_test.csv        # Caractéristiques de test prétraitées
|       |-- y_train.csv       # Variable cible d'entraînement
|       `-- y_test.csv        # Variable cible de test
|-- /notebooks
|   `-- data_exploration.ipynb # Notebook pour l'analyse exploratoire
|-- /scripts
|   |-- preprocess_data.py    # Script de prétraitement des données
|   `-- train_model.py        # Script d'entraînement du modèle
`-- README.md
```

## 2. Comment exécuter

1.  **Clonez le dépôt** :
    ```bash
    git clone git@github.com:LEGBEDJE/Prediction-bitcoin.git
    cd predictionbitcoin
    ```

2.  **Créez un environnement virtuel** (recommandé) :
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Téléchargez les données** :
    Le jeu de données `BTC-USD.csv` doit être placé dans le répertoire `data/raw/`. Vous pouvez le télécharger manuellement depuis des sources comme [Investing.com](https://www.investing.com/crypto/bitcoin/btc-usd-historical-data) ou [CoinMarketCap](https://coinmarketcap.com/currencies/bitcoin/historical-data/). Assurez-vous de nettoyer les colonnes numériques (supprimer les virgules et convertir 'K'/'M' en notation scientifique) si nécessaire, comme indiqué dans le script `preprocess_data.py`.

5.  **Exécutez les scripts** :

    *   Pour prétraiter les données :
        ```bash
        python3 scripts/preprocess_data.py
        ```

    *   Pour entraîner et évaluer le modèle :
        ```bash
        python3 scripts/train_model.py
        ```

## 3. Analyse Exploratoire des Données

Le notebook `notebooks/data_exploration.ipynb` contient l'analyse exploratoire des données. Il visualise l'évolution du prix de clôture du Bitcoin et fournit des statistiques descriptives.

## 4. Prétraitement des Données

Le script `scripts/preprocess_data.py` effectue les opérations suivantes :

*   **Nettoyage des données** : Conversion des colonnes numériques et gestion des caractères spéciaux.
*   **Création de caractéristiques décalées** : Utilisation du prix de clôture du jour précédent (`Close_lag1`) comme caractéristique pour la prédiction afin d'éviter la fuite de données.
*   **Division chronologique des données** : Les données sont divisées en ensembles d'entraînement et de test de manière chronologique (80% pour l'entraînement, 20% pour le test), ce qui est essentiel pour les séries temporelles.
*   **Mise à l'échelle des caractéristiques** : `StandardScaler` est utilisé pour normaliser les caractéristiques.

## 5. Modélisation et Évaluation

Un modèle `RandomForestRegressor` a été entraîné pour prédire le prix de clôture du Bitcoin.

Les performances ont été évaluées à l'aide de deux métriques :
*   **Mean Squared Error (MSE)** : Mesure l'erreur quadratique moyenne entre les valeurs prédites et réelles. Un MSE plus faible est meilleur.
*   **R-squared (R2)** : Représente la proportion de la variance de la variable dépendante qui est prévisible à partir des variables indépendantes. Un R2 plus proche de 1 est meilleur.

### Résultats

| Modèle        | Mean Squared Error (MSE) | R-squared (R2) |
| ------------- | ------------------------ | -------------- |
| Random Forest | 6519023.71               | 0.96           |

Le modèle Random Forest a obtenu un R-squared de 0.96, ce qui indique qu'il explique 96% de la variance du prix de clôture du Bitcoin. Le MSE est relativement élevé en raison de la nature volatile des prix du Bitcoin, mais le R2 élevé montre que le modèle capture bien les tendances.

## 6. Pistes d'Amélioration

Pour améliorer davantage les performances, les étapes suivantes pourraient être envisagées :

*   **Ingénierie des caractéristiques avancée** : Ajouter d'autres caractéristiques décalées (par exemple, prix de clôture des 7 derniers jours), des moyennes mobiles, des indicateurs techniques (RSI, MACD, etc.).
*   **Optimisation des hyperparamètres** : Utiliser `GridSearchCV` ou `RandomizedSearchCV` pour trouver les meilleurs hyperparamètres pour le modèle Random Forest ou d'autres modèles.
*   **Explorer d'autres modèles** : Tester des modèles de séries temporelles spécifiques comme ARIMA, Prophet, ou des réseaux de neurones récurrents (LSTM).
*   **Données supplémentaires** : Intégrer des données macroéconomiques, des actualités ou des sentiments du marché.
