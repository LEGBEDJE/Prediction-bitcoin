
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger les données
df = pd.read_csv('/home/legbedje/Documents/datascienceproject/predictionbitcoin/data/raw/BTC-USD.csv')

# Convertir la colonne 'Date' en datetime
df['Date'] = pd.to_datetime(df['Date'])

# Trier les données par date
df = df.sort_values(by='Date')

# Créer des caractéristiques temporelles
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

# Définir les caractéristiques et la variable cible
features = ['Open', 'High', 'Low', 'Volume', 'year', 'month', 'day', 'dayofweek']
target = 'Close'

X = df[features]
y = df[target]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mettre à l'échelle les caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Enregistrer les données prétraitées
processed_data_path = '/home/legbedje/Documents/datascienceproject/predictionbitcoin/data/processed/'
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(processed_data_path + 'X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(processed_data_path + 'X_test.csv', index=False)
y_train.to_csv(processed_data_path + 'y_train.csv', index=False)
y_test.to_csv(processed_data_path + 'y_test.csv', index=False)

print("Les données ont été prétraitées et enregistrées dans le répertoire data/processed.")
