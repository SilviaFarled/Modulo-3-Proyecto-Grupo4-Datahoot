import funciones as f
import graficos as g
import load_sql as sql

df = f.visualizacion_y_limpieza ('HR RAW DATA.csv')

g.histograma(df)

df= f.gestion_nulos(df, 'HR LIMPIO.csv')

