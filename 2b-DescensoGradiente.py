# Importar las librerías necesarias
import numpy as np
import pandas as pd

# Definir la función de costo o error cuadrático medio(MSE)
def costo(X, y, theta):
  m = len(y) # Número de observaciones
  y_pred = X.dot(theta) # Predicción lineal
  error = (y - y_pred) ** 2 # Error al cuadrado
  return 1 / (2 * m) * np.sum(error) # Costo promedio

# Definir el algoritmo de descenso de gradiente con criterio de diferencia relativa
def descenso_gradiente(X, y, theta, alpha, iteraciones, tolerancia):
  m = len(y) # Número de observaciones
  costo_historico = np.zeros(iteraciones) # Vector para almacenar el costo en cada iteración
  theta_historico = np.zeros((iteraciones, len(theta))) # Matriz para almacenar el valor de theta en cada iteración
  for i in range(iteraciones):
    y_pred = X.dot(theta) # Predicción lineal
    error = np.dot(X.transpose(), (y_pred - y)) # Error ponderado por las variables independientes
    theta = theta - alpha * (1 / m) * error # Actualización de theta
    costo_historico[i] = costo(X, y, theta) # Almacenar el costo
    theta_historico[i,:] = theta.T # Almacenar el valor de theta
    print(f"error de iteracion {i} : {costo(X, y, theta)}")
    if i > 0: # Verificar si hay al menos dos iteraciones
      dif_relativa = abs(costo_historico[i] - costo_historico[i-1]) / costo_historico[i-1] # Calcular la diferencia relativa del costo
      if dif_relativa < tolerancia: # Verificar si la diferencia relativa es menor que la tolerancia
        break # Detener el algoritmo
  return theta, costo_historico, theta_historico


# Leer archivo wines.csv con los datos
df = pd.read_csv('wines.csv')
X = df.iloc[:, :-1]  # Variables independientes
y = df.iloc[:, -1]   # Variable dependiente
m = m = X.shape[0] # Obtener el número de observaciones a partir de X # Número de observaciones


# Agregar una columna de unos a X para termino independiente
X_b = np.c_[np.ones((m, 1)), X]

# Inicializar los parámetros theta con valores aleatorios
theta_inicial = np.zeros(8)

# Definir la tasa de aprendizaje,el número de iteraciones y la tolerancia
alpha = 0.00009
iteraciones = 4400
tolerancia = 0.001

# Aplicar el algoritmo de descenso de gradiente
theta_optimo, costo_historico, theta_historico = descenso_gradiente(X_b, y, theta_inicial, alpha, iteraciones,tolerancia)

# Mostrar el valor óptimo de theta y el costo mínimo
print("Theta óptimo:", theta_optimo)
print("Costo mínimo:", costo_historico[-1])

