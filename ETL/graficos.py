import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

import warnings
warnings.filterwarnings("ignore") 

import scipy.stats as stats
from scipy.stats import shapiro, poisson, chisquare, expon, kstest
from scipy.stats import levene, bartlett, shapiro

import pymysql
from sqlalchemy import create_engine

pd.set_option('display.max_rows', None) # Deslimita visualización de filas
pd.set_option('display.max_columns', None) # Deslimita visualización de columnas


colores_proyecto = [
    'mediumpurple',   # morado medio
    'darkorange',     # naranja fuerte
    'plum',           # morado claro
    'sandybrown',     # naranja suave
    'purple',         # morado intenso
    'lightsalmon',    # naranja rosado
    'orchid',         # morado rosado
    'chocolate'       # naranja oscuro
]


def histograma(df1):
    columnas_number_con_nulos = df1[df1.columns[df1.isnull().any()]].select_dtypes(include = np.number).columns
    for col in list(columnas_number_con_nulos):
        plt.figure(figsize=(8, 5))
        plt.hist(df1[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.show()
        return


def estado_num (df):
    df_num = df.copy()
    df_num['Estado'] = df_num['Estado'].replace({'Desvinculado': 1, 'Activo': 0}).astype(int)
    return df_num



def grafico_1(df1):
    # Comparación de variables emocionales

    variables_emocionales = ['Compromiso', 'Satisf. global', 'Satisf. trabajo', 'Satisf. relaciones','Satisf. conciliación']
    df_emocionales = (df1.groupby('Estado')[variables_emocionales].mean()).round(2).T
    df_emocionales.columns = ["Activo", "Desvinculado"]
    ax = df_emocionales.plot(kind='bar', figsize=(10,6), color=colores_proyecto)
    fig = ax.get_figure() 
    fig.patch.set_facecolor('#dce3f0')      
    ax.set_facecolor('#ffffff')             
    plt.title("Comparación de variables emocionales",fontsize=20, fontweight='bold')
    plt.ylabel("Promedio")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.ylim([1,4])
    plt.show()
    return


def grafico_2(df1):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#dce3f0")
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor("#ffffff")
    columnas = ["Años activo", "Antigüedad", "Años desde ascenso", "Años mismo jefe"]
    axes = axes.flatten()
    for i, col in enumerate(columnas):
        sns.boxplot(data=df1, x="Estado", y=col, ax=axes[i], palette=colores_proyecto)
        axes[i].set_title(f'{col} vs estado', fontsize=22, fontweight='bold')
        axes[i].set_xlabel("Estado", fontsize=14)
        axes[i].set_ylabel(col, fontsize=14)
    for i in [2, 3]:
        axes[i].set_ylim(-1, 18)
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
    return


def grafico_3(df1):
    df_mod = df1.copy()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#dce3f0")  # Cambia el color del fondo general
    ax.set_facecolor("#ffffff") 
    sns.scatterplot(x='Antigüedad', y='Años mismo jefe', data=df_mod, palette=colores_proyecto, marker='v', hue='Estado')
    plt.title("Antigüedad con el mismo responsable",fontsize=20, fontweight='bold')
    plt.ylabel("Años con el mismo responsable",fontsize=14)
    plt.xlabel("Antigüedad en la empresa",fontsize=14)
    plt.legend(title='Estado')
    plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    plt.tight_layout()
    plt.show()
    return


def grafico_4(df1):
    df_num = estado_num(df1)
    df2 = df_num.copy()
    df2 = df2.dropna(subset=['Satisf. conciliación', 'Estado', 'Jornada'])
    df_grouped = df2.groupby(['Satisf. conciliación', 'Jornada'])['Estado'].mean().reset_index()
    df_grouped['% Desvinculados'] = df_grouped['Estado'] * 100
    plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#dce3f0")
    ax.set_facecolor("#ffffff")
    sns.barplot(
        data=df_grouped,
        x='Satisf. conciliación',
        y='% Desvinculados',
        hue='Jornada',
        palette=colores_proyecto)
    plt.title('% Desvinculación según conciliación y tipo de jornada\n', fontsize=14, fontweight='bold')
    plt.ylabel('% de personas desvinculadas')
    plt.xlabel('Satisfacción con la conciliación')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return


def grafico_5(df1):
    # Comparación evaluación desempeño e incremento salarial - no afecta a desvinculado
    df_agrupado = df1.groupby(["Evaluación", "Estado"], as_index=False)["% aumento salario"].mean()
    fig, ax = plt.subplots(figsize=(4, 4))  
    fig.patch.set_facecolor('#dce3f0')       
    ax.set_facecolor('#ffffff')              
    sns.barplot(
        data=df_agrupado,            
        x="Evaluación",               
        y="% aumento salario",       
        hue="Estado",                 
        palette= colores_proyecto)
    plt.title("Comparación evaluación e incremento salarial\n",fontsize=10, fontweight='bold')  
    plt.ylabel("% Incremento salarial")                                 
    plt.xlabel("Evaluación del desempeño")                             
    plt.ylim(10, 25)                                                    
    plt.grid(axis='y', linestyle='--', alpha=0.5)                   
    plt.legend(title="Estado")                                        
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))           
    plt.tight_layout()                                                
    plt.show() 
    return     


def grafico_6(df1):
    # En función del género:
    porcentaje_por_genero = df1['Género'].value_counts(normalize=True) * 100
    fig, _ = plt.subplots(figsize=(4, 4))  
    fig.patch.set_facecolor('#dce3f0')       # Fondo de toda la figura
    plt.pie(
        porcentaje_por_genero,                    # Valores a mostrar
        labels=porcentaje_por_genero.index,      # Etiquetas de cada sector (géneros)
        autopct='%1.1f%%',                       # Mostrar porcentaje con un decimal
        colors=colores_proyecto                   # Paleta de colores definida previamente
    )
    plt.title("Distribución por sexo",fontsize=16, fontweight='bold')
    plt.show()
    return

def grafico_7(df1):
    # En función del género:
    df_filtrado = df1[df1['Estado'] == "Desvinculado"]
    porcentaje_por_genero = df_filtrado['Género'].value_counts(normalize=True) * 100
    fig, _ = plt.subplots(figsize=(4, 4))  
    fig.patch.set_facecolor('#dce3f0')       # Fondo de toda la figura
    plt.pie(porcentaje_por_genero, labels=porcentaje_por_genero.index, autopct='%1.1f%%', colors=colores_proyecto)
    plt.title("Perfil desvinculado: HOMBRE",fontsize=14, fontweight='bold')
    plt.show()
    return


def grafico_8(df1):
    # En función de la edad:
    df_filtrado = df1[df1['Estado'] == "Desvinculado"]
    media_edad = df_filtrado["Edad"].mean()
    fig, ax = plt.subplots(figsize=(4, 5))   
    fig.patch.set_facecolor('#dce3f0')       # Fondo de toda la figura
    ax.set_facecolor('#ffffff')              # Fondo del área del gráfico
    sns.boxplot(y=df_filtrado['Edad'], palette=colores_proyecto, width=0.3)
    plt.axhline(media_edad, color='orange', linestyle='--', linewidth=1.5, label=f'Media: {media_edad:.1f} años')
    plt.title("Perfil desvinculado: EDAD",fontsize=14, fontweight='bold')
    plt.ylabel("Edad")
    plt.tight_layout()
    plt.show()
    return


def grafico_9(df):
    df_num = estado_num(df)
    # Revisión otros valores medios del trabajador desvinculado
    df_num_stats = df_num.query("Estado == 1")[['Años activo', 'Antigüedad', 'Años mismo jefe', 'Categoría', 'Acciones empresa']].describe().T
    fig, ax = plt.subplots(figsize=(6, 4))   
    fig.patch.set_facecolor('#dce3f0')       # Fondo de toda la figura
    ax.set_facecolor('#ffffff')              # Fondo del área del gráfico
    sns.barplot(x=df_num_stats.index, y=df_num_stats["mean"], palette=colores_proyecto)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    x_labels = df_num_stats.index
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, ha='center')
    plt.title("Perfil desvinculado: OTROS VALORES",fontsize=16, fontweight='bold')
    plt.ylabel("Valor promedio")
    plt.xlabel("Variables")
    plt.show()
    return


def grafico_10(df1):
    # INGRESOS no parece afectar a la desvinculación.
    df_ingreso = df1.groupby('Estado')['Ingreso mensual'].mean().round(2)
    x = np.arange(len(df_ingreso))
    etiquetas = df_ingreso.index
    valores = df_ingreso.values
    fig, ax = plt.subplots(figsize=(4, 4))   
    fig.patch.set_facecolor('#dce3f0')       # Fondo de toda la figura
    ax.set_facecolor('#ffffff')              # Fondo del área del gráfico
    plt.bar(x=x, height=valores, color=colores_proyecto, width=0.3)
    plt.xticks(ticks=x, labels=etiquetas)
    plt.xlim(-0.5, len(x) - 0.5)
    plt.xlabel('Estado')
    plt.ylabel('Ingreso mensual')
    plt.title('Estado VS Ingresos',fontsize=16, fontweight='bold')
    plt.show()
    return


def grafico_11(df1):
    # Calcular la moda del ingreso mensual (valor más frecuente)
    moda = df1["Ingreso mensual"].mode().iloc[0]
    df_ingresos = df1[df1["Ingreso mensual"] != moda]
    fig, ax = plt.subplots(figsize=(7, 5))   
    fig.patch.set_facecolor('#dce3f0')       # Fondo de toda la figura
    ax.set_facecolor('#ffffff')              # Fondo del área del gráfico
    sns.boxplot(
        x='Estado', y='Ingreso mensual', hue='Género', data=df_ingresos, 
        palette=colores_proyecto, legend=True, dodge=True, width=0.6
    )
    plt.axhline(moda, color="plum", linestyle='--', linewidth=1.5, label=f'Moda: {moda:.0f} €')
    plt.title("Ingresos por género y estado",fontsize=16, fontweight='bold')
    plt.xlabel('Estado')
    plt.ylabel('Valoración')
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


def grafico_12(df1):
  # MAPA de calor para valorar las variables numéricas que pueden tener relación
    df_mapacalor = df1[['Edad', 'Nivel estudios', 'Trabajos previos', 'Años activo',
                    'Antigüedad', 'Años desde ascenso', 'Años mismo jefe', 'Categoría',
                    'Evaluación', 'Ingreso mensual', '% aumento salario']]
    valores = df_mapacalor.columns
    df_correlation = df1[valores].corr()
    plt.figure(figsize = (8, 8))
    sns.heatmap(df_correlation,  # Dibuja el mapa de calor con las correlaciones
                annot = True,    # Muestra los valores numéricos en cada celda
                fmt = ".2f",     # Formato de los números con 2 decimales
                cmap = "coolwarm",  # Paleta de colores que va de frío a cálido
                vmax = 1,        # Valor máximo en el gradiente de color
                vmin = -1,       # Valor mínimo en el gradiente de color
                )
    plt.show()
    return



def impresion_graficos(df1):
    print("Comparación de variables emocionales")
    grafico_1(df1)
    print("Valoración historia laboral")
    grafico_2(df1)
    print("Búsqueda de relación entre antigüedad, tiempo con el mismo responsable y estado")
    grafico_3(df1)
    print("# Valoración de la conciliación respecto a la desvinculación")
    grafico_4(df1)
    print("Comparación evaluación desempeño e incremento salarial - no afecta a desvinculado")
    grafico_5(df1)
    print("En función del género:1")
    grafico_6(df1)
    print("En función del género:2")
    grafico_7(df1)
    print("En función de la edad")
    grafico_8(df1)
    print("Revisión otros valores medios del trabajador desvinculado")
    grafico_9(df1)
    print("INGRESOS no parece afectar a la desvinculación")
    grafico_10(df1)
    print("Igresos por género y estado")
    grafico_11(df1)
    print("Mapa de correlación")
    grafico_12(df1)
    return