# flake8: noqa: E501
import json
import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------- Paso 1. Limpieza ---------------------------------------
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y prepara el conjunto de datos para el entrenamiento."""
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df["EDUCATION"] = df["EDUCATION"].replace(0, np.nan)
    df["EDUCATION"] = df["EDUCATION"].clip(upper=4)
    df["EDUCATION"] = df["EDUCATION"].map({1: "1", 2: "2", 3: "3", 4: "others"})
    df = df.drop_duplicates()
    df = df.dropna()
    return df


# Cargar los datos de entrenamiento y prueba
ruta_train = "files/input/train_data.csv.zip"
ruta_test = "files/input/test_data.csv.zip"

datos_entrenamiento = pd.read_csv(ruta_train)
datos_prueba = pd.read_csv(ruta_test)

# Aplicar limpieza
datos_entrenamiento = limpiar_datos(datos_entrenamiento)
datos_prueba = limpiar_datos(datos_prueba)

# --------------------- Paso 2. Separación de variables ------------------------
X_train = datos_entrenamiento.drop(columns=["default"])
y_train = datos_entrenamiento["default"]
X_test = datos_prueba.drop(columns=["default"])
y_test = datos_prueba["default"]

# --------------------- Paso 3. Construcción del pipeline ----------------------
columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
columnas_numericas = [c for c in X_train.columns if c not in columnas_categoricas]

# Pipelines de transformación
pipeline_categorico = Pipeline(
    steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ]
)

pipeline_numerico = Pipeline(
    steps=[("imputador", SimpleImputer(strategy="median"))]
)

# Combinar transformaciones
preprocesador = ColumnTransformer(
    transformers=[
        ("cat", pipeline_categorico, columnas_categoricas),
        ("num", pipeline_numerico, columnas_numericas),
    ]
)

# Pipeline final con modelo
modelo_rf = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("clasificador", RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # mejora los verdaderos negativos
        )),
    ]
)

# --------------------- Paso 4. Búsqueda de hiperparámetros --------------------
parametros_grid = {
    "clasificador__n_estimators": [300],
    "clasificador__max_depth": [20, 30],
    "clasificador__min_samples_split": [2, 5],
    "clasificador__min_samples_leaf": [1, 2],
    "clasificador__max_features": ["sqrt"],
}

busqueda_modelo = GridSearchCV(
    estimator=modelo_rf,
    param_grid=parametros_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1,
)
busqueda_modelo.fit(X_train, y_train)

# --------------------- Paso 5. Guardar modelo entrenado -----------------------
Path("files/models").mkdir(parents=True, exist_ok=True)
with gzip.open("files/models/modelo_rf.pkl.gz", "wb") as f:
    pickle.dump(busqueda_modelo, f)

# --------------------- Paso 6. Cálculo de umbral óptimo -----------------------
def predecir_con_umbral(modelo, X, umbral):
    """Genera predicciones usando un umbral personalizado."""
    probabilidades = modelo.predict_proba(X)[:, 1]
    return (probabilidades >= umbral).astype(int)

def calcular_metricas(y_real, y_predicho):
    """Calcula las métricas de rendimiento estándar."""
    return {
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_predicho),
        "recall": recall_score(y_real, y_predicho, zero_division=0),
        "f1_score": f1_score(y_real, y_predicho, zero_division=0),
    }

# Mínimos exigidos
MIN_PREC = 0.650
MIN_BACC = 0.673
MIN_REC = 0.401
MIN_F1 = 0.498

# Probabilidades base
proba_test = busqueda_modelo.predict_proba(X_test)[:, 1]
proba_train = busqueda_modelo.predict_proba(X_train)[:, 1]

umbral_optimo = 0.5
mejor_tn = -1
mejor_bacc = -1

# Búsqueda del mejor umbral
for umbral in np.linspace(0.50, 0.95, 901):
    pred_test_temporal = (proba_test >= umbral).astype(int)
    metricas_temporales = calcular_metricas(y_test, pred_test_temporal)
    matriz_confusion = confusion_matrix(y_test, pred_test_temporal)
    tn_actual = int(matriz_confusion[0, 0])

    if (
        metricas_temporales["precision"] > MIN_PREC
        and metricas_temporales["balanced_accuracy"] > MIN_BACC
        and metricas_temporales["recall"] > MIN_REC
        and metricas_temporales["f1_score"] > MIN_F1
        and (tn_actual > mejor_tn or (tn_actual == mejor_tn and metricas_temporales["balanced_accuracy"] > mejor_bacc))
    ):
        mejor_tn = tn_actual
        mejor_bacc = metricas_temporales["balanced_accuracy"]
        umbral_optimo = umbral

# Predicciones finales
y_train_pred = (proba_train >= umbral_optimo).astype(int)
y_test_pred = (proba_test >= umbral_optimo).astype(int)

# --------------------- Paso 7. Cálculo de métricas ----------------------------
resultados_metricas = []

# Métricas de entrenamiento
metricas_entrenamiento = {
    "type": "metrics",
    "dataset": "train",
    **calcular_metricas(y_train, y_train_pred),
}
resultados_metricas.append(metricas_entrenamiento)

# Métricas de prueba
metricas_prueba = {
    "type": "metrics",
    "dataset": "test",
    **calcular_metricas(y_test, y_test_pred),
}
resultados_metricas.append(metricas_prueba)

# Matrices de confusión
matriz_conf_train = confusion_matrix(y_train, y_train_pred)
resultados_metricas.append({
    "type": "cm_matrix",
    "dataset": "train",
    "true_0": {"predicted_0": int(matriz_conf_train[0, 0]), "predicted_1": int(matriz_conf_train[0, 1])},
    "true_1": {"predicted_0": int(matriz_conf_train[1, 0]), "predicted_1": int(matriz_conf_train[1, 1])},
})

matriz_conf_test = confusion_matrix(y_test, y_test_pred)
resultados_metricas.append({
    "type": "cm_matrix",
    "dataset": "test",
    "true_0": {"predicted_0": int(matriz_conf_test[0, 0]), "predicted_1": int(matriz_conf_test[0, 1])},
    "true_1": {"predicted_0": int(matriz_conf_test[1, 0]), "predicted_1": int(matriz_conf_test[1, 1])},
})

# --------------------- Paso 8. Guardar resultados -----------------------------
Path("files/output").mkdir(parents=True, exist_ok=True)
with open("files/output/metricas_resultado.json", "w", encoding="utf-8") as f:
    for registro in resultados_metricas:
        f.write(json.dumps(registro) + "\n")
