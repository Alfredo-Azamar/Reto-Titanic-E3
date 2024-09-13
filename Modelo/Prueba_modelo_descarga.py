import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import randint, uniform
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

Datos = pd.read_csv("train.csv")
Datos_venta = Datos[["HomePlanet", "Destination", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Transported", "Age","CryoSleep"]]

# Limpieza avanzada
# Características numéricas (ej: 'RoomService', 'FoodCourt', etc.) se llenarán con valores medianos
caracteristicas_numericas = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
for feature in caracteristicas_numericas:
    Datos_venta[feature] = Datos_venta[feature].fillna(Datos_venta[feature].median()).astype(int)

# Llenar valores faltantes para las características categóricas
# 'HomePlanet', 'Destination' y 'VIP' se llenarán con los valores de moda
Datos_venta['HomePlanet'] = Datos_venta['HomePlanet'].fillna(Datos_venta['HomePlanet'].mode()[0])
Datos_venta['HomePlanet'] = Datos_venta['HomePlanet'].replace('Europa','Earth')
Datos_venta['Destination'] = Datos_venta['Destination'].fillna(Datos_venta['Destination'].mode()[0])
Datos_venta['VIP'] = Datos_venta['VIP'].fillna('False')

# Para CryoSleep (si alguien está en criosueño o no),
# Llenar los valores faltantes con False (suponiendo que no están en criosueño)
Datos_venta['CryoSleep'] = Datos_venta['CryoSleep'].fillna('False')

# Transformar columnas booleanas y categóricas en binarias (0/1)
Datos_venta['VIP'] = Datos_venta['VIP'].apply(lambda x: 1 if x == 'True' else 0)
Datos_venta['CryoSleep'] = Datos_venta['CryoSleep'].apply(lambda x: 1 if x == 'True' else 0)

Datos_codificados = pd.get_dummies(Datos_venta, columns=['HomePlanet', 'Destination'], drop_first=True)

#Matriz de correlación
# Solo valore num en col
numerical_columns = Datos_codificados.select_dtypes(include=['number'])
correlation_matrix = numerical_columns.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

#Detección de outliers
def limitar_outliers(df, feature, cuantil_bajo=0.01, cuantil_alto=0.99):
    """ Limitar outliers en una característica basada en cuantiles """
    limite_bajo = df[feature].quantile(cuantil_bajo)
    limite_alto = df[feature].quantile(cuantil_alto)
    df[feature] = np.clip(df[feature], limite_bajo, limite_alto)

for feature in caracteristicas_numericas:
    limitar_outliers(Datos_codificados, feature)

    # Columnas que no sean numéricas
print(Datos_codificados.select_dtypes(include='object').columns)

# One-hot encoding a las columnas no numéricas restantes
# Verificar si queda alguna columna categórica
if not Datos_codificados.select_dtypes(include='object').empty:
    Datos_codificados = pd.get_dummies(Datos_codificados, drop_first=True)


scaler = MinMaxScaler()
X_escalado = pd.DataFrame(scaler.fit_transform(Datos_codificados.drop('Transported', axis=1)),
                          columns=Datos_codificados.columns.drop('Transported'))

x_train, x_test, y_train, y_test = train_test_split(X_escalado, Datos_codificados['Transported'], test_size=0.2)

joblib.dump(scaler,"dt1_scaler_V6.joblib")

print(x_train.shape)


model = CatBoostClassifier(
    iterations=861,         # Número de árboles
    depth=8,                 # Profundidad de los árboles
    learning_rate=0.027175353374651955,      # Tasa de aprendizaje 
    random_strength=7,     # Aleatoriedad en la selección de splits
    border_count=831,        # Número de bordes en la cuantización
    l2_leaf_reg = 9,    
    loss_function='CrossEntropy', # Función de pérdida
    verbose=100              # Imprimir información de progreso cada 100 iteraciones
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mostrar resultados
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'F1: {f1*100:.2f}%')

joblib.dump(model,"dt1_V6.joblib")

test= pd.read_csv("test.csv")
Datos_test = test[["HomePlanet", "Destination", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Age","CryoSleep"]]

# Limpieza avanzada
# Características numéricas (ej: 'RoomService', 'FoodCourt', etc.) se llenarán con valores medianos
caracteristicas_numericas = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
for feature in caracteristicas_numericas:
    Datos_test[feature] = Datos_test[feature].fillna(Datos_test[feature].median()).astype(int)

# Llenar valores faltantes para las características categóricas
# 'HomePlanet', 'Destination' y 'VIP' se llenarán con los valores de moda
Datos_test['HomePlanet'] = Datos_test['HomePlanet'].fillna(Datos_test['HomePlanet'].mode()[0])
Datos_test['Destination'] = Datos_test['Destination'].fillna(Datos_test['Destination'].mode()[0])
Datos_test['VIP'] = Datos_test['VIP'].fillna('False')

# Para CryoSleep (si alguien está en criosueño o no),
# Llenar los valores faltantes con False (suponiendo que no están en criosueño)
Datos_test['CryoSleep'] = Datos_test['CryoSleep'].fillna('False')

# Transformar columnas booleanas y categóricas en binarias (0/1)
Datos_test['VIP'] = Datos_test['VIP'].apply(lambda x: 1 if x == 'True' else 0)
Datos_test['CryoSleep'] = Datos_test['CryoSleep'].apply(lambda x: 1 if x == 'True' else 0)

Datos_codificados = pd.get_dummies(Datos_test, columns=['HomePlanet', 'Destination'], drop_first=True)

#Matriz de correlación
# Solo valore num en col
numerical_columns = Datos_codificados.select_dtypes(include=['number'])
correlation_matrix = numerical_columns.corr()

#Detección de outliers
def limitar_outliers(df, feature, cuantil_bajo=0.01, cuantil_alto=0.99):
    limite_bajo = df[feature].quantile(cuantil_bajo)
    limite_alto = df[feature].quantile(cuantil_alto)
    df[feature] = np.clip(df[feature], limite_bajo, limite_alto)

for feature in caracteristicas_numericas:
    limitar_outliers(Datos_codificados, feature)

    # Columnas que no sean numéricas
print(Datos_codificados.select_dtypes(include='object').columns)

# One-hot encoding a las columnas no numéricas restantes
# Verificar si queda alguna columna categórica
if not Datos_codificados.select_dtypes(include='object').empty:
    Datos_codificados = pd.get_dummies(Datos_codificados, drop_first=True)


scaler = MinMaxScaler()
X_escalado = pd.DataFrame(scaler.fit_transform(Datos_codificados),
                          columns=Datos_codificados.columns)


y_pred_kaggle = model.predict(X_escalado)

# Supongamos que y_pred_kaggle es tu array de predicciones
# Convertir el array de predicciones a strings
y_pred_strings = np.where(y_pred_kaggle == 1, 'True', 'False')

# Convertir los IDs a strings
ids_de_pasajeros = test['PassengerId'].astype(str)

# Crear un DataFrame con los IDs como cadenas y las predicciones en formato de string
predicciones_df = pd.DataFrame({
    'PassengerId': ids_de_pasajeros,
    'Transported': y_pred_strings
})

# Guardar el DataFrame en un archivo CSV
predicciones_df.to_csv('predicciones_V6.csv', index=False)
