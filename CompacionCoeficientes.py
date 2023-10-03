import pandas as pd

def predecir_calidad_vino(fixed_acidity, citric_acid, residual_sugar, sulfur_dioxide, density, pH, alcohol):
    # Coeficientes finales obtenidos durante el entrenamiento
    #coeficientes = [1.7543705,0.2288474,0.9684832,0.00362239,-0.00419633,-0.98809291,0.93700276,0.07920933]
    #coeficientes = [202.57040640879495,0.12074881931618718,0.4022856670614374,0.0967274322647311,
    #                0.0008832983891760016, -204.2263526334898, 1.0753218440914794, 0.11436853518947865]
    coeficientes = [0.02948124, 0.16919393, 0.00821481, 0.02149984, 0.00160816, 0.02912079,0.0966908,  0.38042342]
    #coeficientes = [1.7537438,0.32436092, 0.97484543,  1.84712528, -0.08664456, -0.98768211 ,0.9193967,  -0.22196713]
    #coeficientes = [ 1.75757476,0.28576758,0.97161389 , 0.0280822  ,-0.00672289 ,-0.98481065 ,0.946094  ,  0.05794842]
    
    # Calcula la calidad con los coeficientes
    calidad = coeficientes[0] + \
              coeficientes[1] * fixed_acidity + \
              coeficientes[2] * citric_acid + \
              coeficientes[3] * residual_sugar + \
              coeficientes[4] * sulfur_dioxide + \
              coeficientes[5] * density + \
              coeficientes[6] * pH + \
              coeficientes[7] * alcohol
    
    return calidad

# Cargar los datos desde el archivo CSV
df = pd.read_csv('wines.csv')

# Obtener las primeras 100 filas del conjunto de datos
primeras_10_filas = df.head(100)

# Iterar sobre las filas
for index, fila in primeras_10_filas.iterrows():
    # Obtener los valores de las variables independientes de la fila
    fixed_acidity = fila['fixed acidity']
    citric_acid = fila['citric acid']
    residual_sugar = fila['residual sugar']
    sulfur_dioxide = fila['sulfur dioxide']
    density = fila['density']
    pH = fila['pH']
    alcohol = fila['alcohol']
    
    # Llama a la función para predecir la calidad del vino
    calidad_predicha = predecir_calidad_vino(fixed_acidity, citric_acid, residual_sugar, sulfur_dioxide, density, pH, alcohol)
    
    # Obtiene la calidad original de la fila
    calidad_original = fila['quality']
    
    # Imprime los resultados para cada fila
    print(f"Fila {index + 1} - Calidad del vino predicha: {calidad_predicha}, Calidad original del vino: {calidad_original}")
