
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données prétraitées
processed_data_path = '/home/legbedje/Documents/datascienceproject/predictionbitcoin/data/processed/'
X_train = pd.read_csv(processed_data_path + 'X_train.csv')
X_test = pd.read_csv(processed_data_path + 'X_test.csv')
y_train = pd.read_csv(processed_data_path + 'y_train.csv').values.ravel()
y_test = pd.read_csv(processed_data_path + 'y_test.csv').values.ravel()

# Entraîner le modèle de régression Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Résultats du modèle de Régression Random Forest ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
