import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger les données
df = pd.read_csv('/home/legbedje/Documents/datascienceproject/predictionbitcoin/data/raw/BTC-USD.csv')

# Nettoyage des données
df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
df['Open'] = df['Open'].astype(str).str.replace(',', '').astype(float)
df['High'] = df['High'].astype(str).str.replace(',', '').astype(float)
df['Low'] = df['Low'].astype(str).str.replace(',', '').astype(float)

df['Vol.'] = df['Vol.'].astype(str).str.replace('K', 'e3').str.replace('M', 'e6').str.replace(',', '')
df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce')

df['Change %'] = df['Change %'].astype(str).str.replace('%', '')
df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')

# Renommer la colonne 'Price' en 'Close'
df = df.rename(columns={'Price': 'Close'})

# Trier les données par date
df = df.sort_values(by='Date')

# Créer des caractéristiques décalées (lagged features)
df['Close_lag1'] = df['Close'].shift(1)

# Gérer les valeurs manquantes (introduites par errors='coerce' et shift)
df = df.dropna()

# Créer des caractéristiques temporelles
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

# Définir les caractéristiques et la variable cible
# Nous utilisons Close_lag1 au lieu de Open, High, Low du même jour
features = ['Close_lag1', 'Vol.', 'year', 'month', 'day', 'dayofweek']
target = 'Close'

X = df[features]
y = df[target]

# Diviser les données en ensembles d'entraînement et de test
# Pour les séries temporelles, il est préférable de diviser chronologiquement
train_size = int(len(df) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(df)]
y_train, y_test = y[0:train_size], y[train_size:len(df)]

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