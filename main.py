# Importar librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Fase I: Definición de necesidades
def load_data(csv_path):
    """
    Carga los datos desde un archivo CSV.
    Args:
        csv_path (str): Ruta del archivo CSV.
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(csv_path)

# Fase II: Estudio y comprensión de los datos
def summarize_data(df):
    """
    Muestra estadísticas y verifica valores faltantes.
    Args:
        df (pd.DataFrame): DataFrame de entrada.
    """
    print("Resumen de los datos:\n", df.describe())
    print("\nValores faltantes:\n", df.isnull().sum())

# Fase III: Análisis exploratorio de datos (EDA)
def eda(df):
    """
    Realiza análisis exploratorio con gráficos básicos.
    Args:
        df (pd.DataFrame): DataFrame de entrada.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment_level'], kde=True)
    plt.title('Distribución del Nivel de Sentimiento')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='department', y='sentiment_level')
    plt.title('Nivel de Sentimiento por Departamento')
    plt.show()

    # Matriz de correlación
    correlation_matrix = df[['employee_id', 'sentiment_level', 'bad_call']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
    plt.title('Matriz de Correlación')
    plt.show()
    
    # Mostrar departamentos y empleados con más llamadas malas
    print("\nDepartamentos con más llamadas malas:")
    print(df[df['bad_call'] == 1]['department'].value_counts())

    print("\nEmpleados con más llamadas malas:")
    print(df[df['bad_call'] == 1]['employee_id'].value_counts())



def correlation_analysis(df):
    """
    Calcula y visualiza la matriz de correlación entre las variables numéricas.
    Args:
        df (pd.DataFrame): DataFrame de entrada.
    """
    # Calcular matriz de correlación
    corr = df[['employee_id', 'sentiment_level', 'bad_call']].corr()

    print("\nMatriz de correlación:\n", corr)

    # Visualización de la matriz de correlación
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Mapa de Calor - Matriz de Correlación')
    plt.show()

# Fase IV: Modelado
def apply_clustering(df, n_clusters=3):
    """
    Aplica clustering K-means basado en el nivel de sentimiento.
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        n_clusters (int): Número de clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['sentiment_level']])
    print("Centroides de los clusters:\n", kmeans.cluster_centers_)

    # Graficar los clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sentiment_level', y='bad_call', hue='cluster', palette='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], [0] * n_clusters, s=100, c='red', marker='X', label='Centroides')
    plt.title('Clustering de Nivel de Sentimiento')
    plt.legend()
    plt.show()

def apply_classification(df, max_depth=3):
    """
    Entrena un árbol de decisión para predecir llamadas malas y muestra su visualización.
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        max_depth (int): Profundidad máxima del árbol para mejorar la visualización.
    """
    # Codificar variables categóricas
    df_encoded = pd.get_dummies(df, columns=['department'], drop_first=True)

    # Definir variables independientes (X) y dependiente (y)
    X = df_encoded.drop(columns=['bad_call'])
    y = df_encoded['bad_call']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrenar el árbol de decisión con profundidad limitada
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = clf.predict(X_test)

    # Evaluar el modelo
    evaluate_model(y_test, y_pred)

    # Visualización del árbol de decisión
    plt.figure(figsize=(24, 12))
    plot_tree(
        clf,
        filled=True,
        feature_names=X.columns,
        class_names=['Good', 'Bad'],
        fontsize=12,
        rounded=True,
        precision=2
    )
    plt.show()

# Fase V: Evaluación
def evaluate_model(y_true, y_pred):
    """
    Evalúa el modelo utilizando métricas de clasificación.
    Args:
        y_true (array): Etiquetas verdaderas.
        y_pred (array): Etiquetas predichas.
    """
    print("Matriz de Confusión:\n", confusion_matrix(y_true, y_pred))
    print("\nReporte de Clasificación:\n", classification_report(y_true, y_pred))

# Fase VI: Despliegue
def simulate_deployment(df):
    """
    Simula el despliegue del sistema mediante informes automatizados.
    Args:
        df (pd.DataFrame): DataFrame de entrada.
    """
    print("Simulación de despliegue: Generación de insights automatizados.")

# Ejecución del análisis
if __name__ == "__main__":
    # Ruta del CSV (cambiar según tu archivo)
    csv_path = "data/call_center_data.csv"

    # Cargar los datos
    df = load_data(csv_path)

    # Resumen de los datos
    summarize_data(df)

    # Análisis exploratorio
    eda(df)

    # Aplicar clustering
    apply_clustering(df)

    # Aplicar clasificación
    apply_classification(df)

    # Simulación de despliegue
    simulate_deployment(df)
