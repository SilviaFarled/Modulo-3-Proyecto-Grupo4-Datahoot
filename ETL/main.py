import funciones as f
import graficos as g
import load_sql as sql

# Apertura de fichero, limpieza y normalización de columnas
df = f.visualizacion_y_limpieza ('HR RAW DATA.csv')

# Histograma de todas las columnas numéricas para decidir las imputaciones
g.histograma(df)

# Gestión de nulos, guardado y EDA sobre el archivo final para valorar la limpieza
df= f.gestion_nulos(df, 'HR LIMPIO.csv')

# Prueba de hipótesis que devuelve sólo las variables interesantes
f.realizar_prueba_hipotesis(df)

# Impresión de los gráficos configurados
g.impresion_graficos(df)

# Creación de la base de datos y carga del Data Frame
sql.data_upload('Employees', df)