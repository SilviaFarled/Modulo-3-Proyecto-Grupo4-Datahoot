#Importaciones

import pandas as pd
import numpy as np 
from IPython.display import display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

# Variables

nuevas_columnas = {
    "Age": "edad",
    "Attrition": "desvinculado",
    "BusinessTravel": "frecuencia_viaje",
    "DailyRate": "tarifa_diaria",
    "Department": "departamento",
    "DistanceFromHome": "distancia_domicilio",
    "Education": "nivel_estudios",
    "EducationField": "campo_estudios",
    "employeecount": "employeecount",
    "employeenumber": "numero_empleado",
    "EnvironmentSatisfaction": "nivel_satisfaccion_global",
    "Gender": "genero",
    "HourlyRate": "tarifa_hora",
    "JobInvolvement": "nivel_compromiso",
    "JobLevel": "categoria",
    "JobRole": "puesto",
    "JobSatisfaction": "nivel_satisfaccion_trabajo",
    "MaritalStatus": "estado_civil",
    "MonthlyIncome": "ingreso_mensual",
    "MonthlyRate": "tarifa_mensual",
    "NUMCOMPANIESWORKED": "trabajos_anteriores",
    "Over18": "over18",
    "OverTime": "horas_extras",
    "PercentSalaryHike": "%_incremento_salarial",
    "PerformanceRating": "evaluacion_desempeño",
    "RelationshipSatisfaction": "nivel_satisfaccion_relaciones",
    "StandardHours": "tipo_jornada",
    "StockOptionLevel": "acceso_acciones_empresa",
    "TOTALWORKINGYEARS": "años_en_activo",
    "TrainingTimesLastYear": "formaciones_ultimo_año",
    "WORKLIFEBALANCE": "nivel_satisfaccion_conciliacion",
    "YearsAtCompany": "antigüedad_empresa",
    "YearsInCurrentRole": "años_puesto_actual",
    "YearsSinceLastPromotion": "años_ultimo_ascenso",
    "YEARSWITHCURRMANAGER": "años_mismo_responsable",
    "SameAsMonthlyIncome": "recibe_mismo_salario",
    "DateBirth": "año_nacimiento",
    "Salary": "salario",
    "RoleDepartament": "rol_y_departamento",
    "NUMBERCHILDREN": "hijos",
    "RemoteWork": "teletrabajo"
}
columnas_orden= [
    "numero_empleado",
    "desvinculado",
    "genero",
    "edad",
    "año_nacimiento",
    "estado_civil",
    "nivel_estudios",
    "campo_estudios",
    "frecuencia_viaje",
    "distancia_domicilio",
    "teletrabajo",
    "tipo_jornada",
    "formaciones_ultimo_año",
    "trabajos_anteriores",
    "años_en_activo",
    "antigüedad_empresa",
    "años_ultimo_ascenso",
    "años_mismo_responsable",
    "años_puesto_actual",
    "departamento",
    "rol_y_departamento",
    "categoria",
    "puesto",
    "horas_extras",
    "evaluacion_desempeño",
    "tarifa_hora",
    "tarifa_diaria",
    "tarifa_mensual",
    "ingreso_mensual",
    "%_incremento_salarial",
    "recibe_mismo_salario",
    "acceso_acciones_empresa",
    "nivel_compromiso",
    "nivel_satisfaccion_global",
    "nivel_satisfaccion_trabajo",
    "nivel_satisfaccion_relaciones",
    "nivel_satisfaccion_conciliacion",
    "employeecount",
    "over18",
    "salario",
    "hijos"
    ]
columnas_eliminar= ["employeecount",
                    "over18",
                    "salario",
                    "hijos",
                    "años_puesto_actual",
                    "rol_y_departamento",
                    "recibe_mismo_salario"]

numeros_en_letras = {
    'fifty-eight': 58,
    'fifty-five': 55,
    'fifty-two': 52,
    'forty-seven': 47,
    'thirty': 30,
    'thirty-one': 31,
    'thirty-seven': 37,
    'thirty-six': 36,
    'thirty-two': 32,
    'twenty-four': 24,
    'twenty-six': 26
}

lista_float = ['tarifa_diaria', 'ingreso_mensual']

lista_int = ['años_en_activo', 'evaluacion_desempeño', 'nivel_satisfaccion_conciliacion', 'numero_empleado']

diccionarios_traduccion = {
    'desvinculado': {
        'No': 'Activo',
        'Yes': 'Desvinculado'
    },
    'genero': {
        'male': 'Hombre',
        'female': 'Mujer'
    },
    'estado_civil': {
        'ns/nc': 'NS/NC',
        'married': 'Casado/a',
        'divorced': 'Divorciado/a',
        'single': 'Soltero/a'
    },
    'campo_estudios': {
        'ns/nc': 'Otro',
        'Life Sciences': 'Ciencias de la vida',
        'Technical Degree': 'Técnico/a',
        'Medical': 'Medicina',
        'Other': 'Otro',
        'Marketing': 'Marketing',
        'Human Resources': 'Recursos Humanos'
    },
    'frecuencia_viaje': {
        'non travel': 'No viaja',
        'travel rarely': 'Viaja raramente',
        'travel frequently': 'Viaja frecuentemente'
    },
    'teletrabajo': {
        'no': 'No',
        'yes': 'Sí'
    },
    'tipo_jornada': {
        'full time': 'Jornada completa',
        'part time': 'Media jornada'
    },
    'departamento': {
        'sin asignar': 'Sin asignar',
        'research & development': 'Investigación y desarrollo',
        'sales': 'Ventas',
        'human resources': 'Recursos Humanos'
    },
    'puesto': {
        'research director': 'Director/a de investigación',
        'manager': 'Gerente',
        'sales executive': 'Ejecutivo/a de ventas',
        'manufacturing director': 'Director/a de producción',
        'research scientist': 'Científico/a de investigación',
        'healthcare representative': 'Representante de salud',
        'laboratory technician': 'Técnico/a de laboratorio',
        'sales representative': 'Representante de ventas',
        'human resources': 'Recursos Humanos'
    },
    'horas_extras': {
        'No': 'No',
        'Yes': 'Sí'
    }
}

renombrar_columnas = {
    'numero_empleado': 'ID',
    'desvinculado': 'Estado',
    'genero': 'Género',
    'edad': 'Edad',
    'año_nacimiento': 'Año nac.',
    'estado_civil': 'Estado civil',
    'nivel_estudios': 'Nivel estudios',
    'campo_estudios': 'Área estudios',
    'frecuencia_viaje': 'Frecuencia viaje',
    'distancia_domicilio': 'Distancia casa',
    'teletrabajo': 'Teletrabajo',
    'tipo_jornada': 'Jornada',
    'formaciones_ultimo_año': 'Formaciones (últ. año)',
    'trabajos_anteriores': 'Trabajos previos',
    'años_en_activo': 'Años activo',
    'antigüedad_empresa': 'Antigüedad',
    'años_ultimo_ascenso': 'Años desde ascenso',
    'años_mismo_responsable': 'Años mismo jefe',
    'departamento': 'Departamento',
    'categoria': 'Categoría',
    'puesto': 'Puesto',
    'horas_extras': 'Horas extra',
    'evaluacion_desempeño': 'Evaluación',
    'tarifa_hora': '€/hora',
    'tarifa_diaria': '€/día',
    'tarifa_mensual': '€/mes',
    'ingreso_mensual': 'Ingreso mensual',
    '%_incremento_salarial': '% aumento salario',
    'acceso_acciones_empresa': 'Acciones empresa',
    'nivel_compromiso': 'Compromiso',
    'nivel_satisfaccion_global': 'Satisf. global',
    'nivel_satisfaccion_trabajo': 'Satisf. trabajo',
    'nivel_satisfaccion_relaciones': 'Satisf. relaciones',
    'nivel_satisfaccion_conciliacion': 'Satisf. conciliación'
}


#Funciones

def lectura(csv):
    df1=pd.read_csv(csv, index_col=0)
    return df1


def eda_basico(df):
    print("🔍 Primeras filas del DataFrame:")
    display(df.head())
    print("📐 Dimensiones:")
    print(df.shape , "\n")
    print("🧠 Información general:")
    display(df.info())
    print("📊 Valores distintos por columna")
    for column in df.columns:
        display(df[column].value_counts())
    print('🌑Nombre de las columnas:')
    display(df.columns)
    print("📊 Tipos de datos por columna:")
    print(df.dtypes, "\n")
    print("📉 Descripción de columnas numéricas:")
    display(df.describe())
    print("🔤 Descripción de columnas categóricas:")
    display(df.describe(include=['O']))
    print("🚫 Valores nulos por columna:")
    display(df.isnull().sum())
    print("📎 Filas duplicadas:")
    dup_count = df.duplicated().sum()
    print(f"Duplicadas: {dup_count}")
    if dup_count > 0:
        print("Ejemplo de duplicados:")
        print(df[df.duplicated()].head(), "\n")
    else:
        print("No hay filas duplicadas.\n")

def rename(df1):
    df1.rename(columns = nuevas_columnas, inplace = True)
    df1= df1[columnas_orden]
    return df1

def eliminar_columnas(df1):
    df1.drop(columnas_eliminar, axis = 1, inplace = True)
    return df1

def reemplazo_edad (df1):
    df1["edad"] = df1["edad"].replace(numeros_en_letras)
    return df1

def pasar_a_float (columna):
        try: 
                return float(columna.replace("$", "").replace(".", "").replace(",", ".").strip())
        except: 
               return np.nan
        
def correccion_tarifa_hora(df1):
    df1['tarifa_hora']=df1['tarifa_hora'].replace("Not Available", None)
    df1['tarifa_hora']=df1['tarifa_hora'].astype(float)
    df1['tarifa_hora'].isnull().sum()
    return df1

def correccion_distancia_domicilio(df1):
    df1["distancia_domicilio"] = ((df1["distancia_domicilio"].abs().astype(float))*1.60934).round(2)
    return df1


def minusculas(df1):
    lista_minusculas= ['departamento','puesto']
    for i in lista_minusculas:
        df1[i] = df1[i].str.lower().str.strip()
    return df1


def pasar_a_int(valor):
    try:
        if isinstance(valor, str):
            valor = valor.replace(",0", "")
        return int(float(valor))  # por si viene como '3,0' o 3.0
    except:
        return np.nan
    
def correccion_genero(df1):
    diccionario_genero = {0:'male', 1:'female'}
    df1['genero'] = df1['genero'].astype(object).map(diccionario_genero)
    return df1

def correccion_estado_civil(df1):
    df1['estado_civil'] = df1['estado_civil'].str.lower().str.replace('marreid','married')
    return df1

def correccion_teletrabajo(df1):
    diccionario_teletrabajo= {'0':'no', '1':'yes','False':'no','True':'yes','Yes':'yes'}
    df1['teletrabajo']=df1['teletrabajo'].map(diccionario_teletrabajo)
    return df1

def correccion_tipo_jornada(df1):
    df1['tipo_jornada']=df1['tipo_jornada'].str.replace('80,0','part time')
    return df1

def correccion_satisf(df1):
    df1.loc[df1['nivel_satisfaccion_global'] > 9, 'nivel_satisfaccion_global'] = df1['nivel_satisfaccion_global'] // 10
    return df1

def correccion_viaje(df1):
    df1['frecuencia_viaje'] =df1['frecuencia_viaje'].str.replace('-',' ').str.replace('_',' ')
    return df1


def gestionar_nulos(df1):
    df1['frecuencia_viaje'] = df1['frecuencia_viaje'].fillna('non travel')
    df1['tipo_jornada'] = df1['tipo_jornada'].fillna('full time')
    null_percent = df1.isnull().mean() * 100
    numeric_cols = df1.select_dtypes(include=['number']).columns
    categorical_cols = df1.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df1.select_dtypes(include=['datetime']).columns
    nulls_numeric = null_percent[numeric_cols]
    nulls_categorical = null_percent[categorical_cols]
    nulls_datetime = null_percent[datetime_cols]
    print(":gráfico_de_barras: Nulos en columnas numéricas:")
    print(nulls_numeric.sort_values(ascending=False))
    print("\n:abc: Nulos en columnas categóricas:")
    print(nulls_categorical.sort_values(ascending=False))
    print("\n:calendario_de_sobremesa: Nulos en columnas de fecha:")
    print(nulls_datetime.sort_values(ascending=False))
    for col, null_pct in nulls_numeric.items():
        if null_pct > 40:
            df1.drop(columns=[col], inplace=True)
        elif np.isclose(df1[col].mean(), df1[col].median()):
            df1[col] = df1[col].fillna(df1[col].mean())
        else:
            df1[col] = df1[col].fillna(df1[col].median())
    for col, null_pct in nulls_categorical.items():
        if null_pct < 10:
            value_counts = df1[col].value_counts(normalize=True, dropna=True)
            if len(value_counts) >= 2:
                top1 = value_counts.iloc[0]
                top2 = value_counts.iloc[1]
                if (top1 - top2) >= 0.20:
                    moda = value_counts.index[0]
                    df1[col] = df1[col].fillna(moda)
                else:
                    df1[col] = df1[col].fillna("Desconocido")
            else:
                moda = value_counts.index[0]
                df1[col] = df1[col].fillna(moda)
    for col, null_pct in nulls_datetime.items():
        if null_pct > 40:
            df1.drop(columns=[col], inplace=True)
        else:
            df1[col] = df1[col].fillna(pd.to_datetime("1900-01-01"))
    return df1

def ajuste_tipo_dato(df1):
    df1[['distancia_domicilio','numero_empleado', 'evaluacion_desempeño','nivel_satisfaccion_conciliacion','años_en_activo']] = df1[['distancia_domicilio','numero_empleado', 'evaluacion_desempeño','nivel_satisfaccion_conciliacion','años_en_activo']].astype(int)
    df1[['tarifa_mensual','tarifa_hora','tarifa_diaria','ingreso_mensual']] = df1[['tarifa_mensual','tarifa_hora','tarifa_diaria','ingreso_mensual']].astype(float)
    return df1

def renombrado2(df1):
    for columna, traduccion in diccionarios_traduccion.items():
        if columna in df1.columns:
            df1[columna] = df1[columna].map(traduccion) 
    df1.rename(columns=renombrar_columnas, inplace=True)
    return df1

def guardado(df1, nombre):
    df1.to_csv(nombre)
    return


#Funciones combinadas

def visualizacion_y_limpieza (csv):
    df1=lectura(csv)
    eda_basico(df1)
    df1=rename(df1)
    df1= eliminar_columnas(df1)
    df1= reemplazo_edad (df1)
    for i in lista_float:
        df1[i] = df1[i].apply(pasar_a_float)
    df1= correccion_tarifa_hora(df1)
    df1= correccion_distancia_domicilio(df1)
    df1= minusculas(df1)
    for col in lista_int:
        df1[col] = df1[col].apply(pasar_a_int).astype("Int64")    
    df1= correccion_genero(df1)
    df1= correccion_estado_civil(df1)
    df1= correccion_teletrabajo(df1)  
    df1= correccion_tipo_jornada(df1) 
    df1= correccion_satisf(df1)
    df1=correccion_viaje(df1)
    return df1

def gestion_nulos(df1, nombre):
    df1= gestionar_nulos(df1)
    df1= ajuste_tipo_dato(df1)
    df1= renombrado2(df1)
    guardado(df1, nombre)
    return df1