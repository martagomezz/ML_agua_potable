# Importamos las librerías

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import pickle

# Cargamos y limpiamos los datos

df = pd.read_csv('data/raw/water_potability.csv')
df.dropna(inplace=True)
df = shuffle(df)

# Nombramos el target y las features

X = df.drop('Potability', axis=1)
y = df['Potability']

# Hacemos el split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Construimos los modelos

clf = GaussianNB()

random_forest = RandomForestClassifier(max_depth=3, max_features = 'sqrt', n_estimators = 600)

xgb = XGBRFClassifier(learning_rate = 0.0001, max_depth=7, n_estimators=300)


# Entrenamos los modelos

modelos = {"clf": clf,
        "rand_forest": random_forest,
        "xgb": xgb}

for name, modelo in modelos.items():
    modelo.fit(X_train, y_train)

# Evaluamos los modelos y los clasificamos

scores = [(i, precision_score(y_test,j.predict(X_test))) for i, j in modelos.items()]

scores = pd.DataFrame(scores, columns=["Model", "Score"]).sort_values(by="Score", ascending=False)
print(scores)

# evaluamos el modelo final

acierto = accuracy_score(y_test, random_forest.predict(X_test))
error = 1 - acierto

print("Acierto:", round(acierto*100, 2), "%")
print("Error:", round(error*100, 2), "%")

c_matrix = confusion_matrix(y_test, random_forest.predict(X_test))
print(c_matrix)

sns.set(rc = {'figure.figsize':(5,3)})
sns.heatmap(c_matrix/np.sum(c_matrix), annot=True, 
            fmt='.2%', cmap='Blues');

# guardamos el nuevo modelo elegido

filename = 'model/final/best_model'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(random_forest, archivo_salida) 