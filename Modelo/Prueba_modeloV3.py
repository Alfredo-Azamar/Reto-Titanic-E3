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

#plt.figure(figsize=(12, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
#plt.show()

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

joblib.dump(scaler,"dt1.joblib_scaler")

print(x_train.shape)

def realizar_random_search(x_train, y_train):
    # Inicializar el modelo de CatBoostClassifier
    catboost_model = CatBoostClassifier(verbose=0)

    param_distributions = {
        'iterations': np.arange(900, 1500, 100),  # Número de iteraciones (cantidad de árboles en el modelo)
        'learning_rate': np.arange(0.01, 0.2, 0.05),  # Tasa de aprendizaje para ajustar el impacto de cada árbol.
        'depth': np.arange(4, 12, 1),  # Profundidad máxima de los árboles. 
        'l2_leaf_reg': np.arange(4, 8, 1),  # Regularización L2 para evitar sobreajuste. 
        #'bagging_temperature': np.arange(0.0, 1.1, 0.1),  # Controla la aleatoriedad del muestreo de datos.
        'random_strength': np.arange(0.5, 10.0, 1.0),  # Controla la aleatoriedad de los valores de las divisiones de los árboles. 
        #'colsample_bylevel': np.arange(0.5, 0.9, 0.1),  # Proporción de características a utilizar en cada nivel del árbol. 
        #'subsample': np.arange(0.5, 0.9, 0.1),  # Fracción de muestras utilizadas en cada iteración. 
        #'scale_pos_weight': np.arange(0.7, 1.4, 0.1),  # Ponderación para manejar desbalance de clases. 
        'border_count': np.arange(500, 800, 50),  # Número de divisiones usadas para convertir características continuas en categóricas. 
    }

    # Configurar la búsqueda de hiperparámetros con validación cruzada
    random_search = RandomizedSearchCV(estimator=catboost_model, param_distributions=param_distributions, cv=5, scoring='accuracy', n_iter=300, verbose=1)

    # Ajustar el modelo a los datos de entrenamiento
    random_search.fit(x_train, y_train)

    # Obtener los mejores hiperparámetros
    best_params = random_search.best_params_
    print("Mejores hiperparámetros encontrados:")
    print(best_params)

    # Devolver los mejores parámetros
    return best_params

# Realizar la búsqueda de los mejores hiperparámetros
mejores_hiperparametros = realizar_random_search(x_train, y_train)

# Inicializar y entrenar CatBoostClassifier con los mejores hiperparámetros
catboost_model = CatBoostClassifier(**mejores_hiperparametros, verbose=0)
catboost_model.fit(x_train, y_train)

# Predecir y evaluar el rendimiento
y_pred_catboost = catboost_model.predict(x_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred_catboost)
precision = precision_score(y_test, y_pred_catboost)
recall = recall_score(y_test, y_pred_catboost)
f1 = f1_score(y_test, y_pred_catboost)

# Mostrar resultados
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'F1: {f1*100:.2f}%')

joblib.dump(scaler,"dt1.joblib_200")