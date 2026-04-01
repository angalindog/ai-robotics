# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "tensorflow",
#   "scikit-learn",
#   "matplotlib"
# ]
# ///

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("./assignments/doc-05/box_office_dataset.csv")

def tipo_exito(row):
    if row["domestic_percentage"] > 60:
        return "Local"
    elif row["foreign_percentage"] > 60:
        return "Extranjero"
    else:
        return "Global"

data["tipo_exito"] = data.apply(tipo_exito, axis=1)

print("\nDistribución tipo de éxito:")
print(data["tipo_exito"].value_counts())

data["tipo_exito"].value_counts().plot(kind='bar')
plt.title("Tipo de éxito de películas")
plt.show()

data["success"] = (data["worldwide_lifetime_gross"] > 500000000).astype(int)

data_model = data.drop(["title", "rank", "worldwide_lifetime_gross"], axis=1)

# OPCIÓN 1: Clasificación binaria
X = data_model.drop(["success", "tipo_exito"], axis=1)
Y = data_model["success"]

# OPCIÓN 2: Clasificación multiclase
# label_encoder = LabelEncoder()
# data_model["tipo_exito_num"] = label_encoder.fit_transform(data_model["tipo_exito"])
# X = data_model.drop(["success", "tipo_exito", "tipo_exito_num"], axis=1)
# Y = data_model["tipo_exito_num"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # para binario
])

# Con multiclase:
# keras.layers.Dense(3, activation='softmax')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=20)

loss, acc = model.evaluate(X_test, Y_test)
print("\nPrecisión:", acc)

pred = model.predict(X_test[:5])

print("\nPredicciones:")
for i in range(5):
    print("Predicción:", pred[i][0], "->", round(pred[i][0]))
    print("Real:", Y_test.iloc[i])
    print("-----")

print("\nPesos del modelo:")
print(model.get_weights())

print("\nPrimeras filas del dataset:")
print(data.head(10))