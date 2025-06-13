# Proyecto de Optimización de Talento - Módulo 3

## 📋 Descripción del Proyecto

Este proyecto de análisis de datos de Recursos Humanos tiene como objetivo **averiguar por qué se han marchado los empleados de la empresa** y proporcionar insights valiosos para mejorar la retención del talento.

El análisis se basa en un dataset de recursos humanos que contiene información sobre empleados activos y desvinculados, incluyendo variables demográficas, laborales, de satisfacción y compensación.

## 🎯 Objetivos

- Identificar los factores principales que influyen en la desvinculación de empleados
- Analizar patrones y correlaciones en los datos de recursos humanos
- Crear visualizaciones que faciliten la comprensión de los hallazgos
- Proporcionar recomendaciones basadas en datos para mejorar la retención del talento

## 📁 Estructura del Proyecto

### Datasets

- **`HR RAW DATA.csv`**: Dataset original sin procesar
- **`HR_RAW_v2.csv`**: Dataset después de la limpieza básica
- **`HR_RAW_v3.csv`**: **Dataset final procesado** con todas las transformaciones aplicadas
- **`Tabla columnas.png`**: Imagen de referencia con información sobre las columnas

### Notebooks de Análisis

#### 📊 PARTE 1: Análisis Exploratorio de Datos (EDA)
**Archivo:** `PARTE-1_project-da-promo-52-modulo-3-team-4-EDA.ipynb`

**Contenido:**
- Exploración inicial del dataset original
- Análisis de la estructura y dimensiones de los datos
- Identificación de tipos de datos y valores únicos
- Detección de problemas en los datos (valores nulos, inconsistencias, etc.)
- Función personalizada `eda_basico()` para análisis sistemático

**Principales hallazgos:**
- Dataset con 1,614 registros y 41 columnas
- Múltiples problemas de calidad de datos identificados
- Necesidad de normalización y limpieza extensiva

#### 🧹 PARTE 2: Limpieza de Datos
**Archivo:** `PARTE-2_project-da-promo-52-modulo-3-team-4-LIMPIEZA.ipynb`

**Contenido:**
- Renombrado de columnas al español con nomenclatura normalizada
- Eliminación de columnas redundantes o innecesarias (7 columnas eliminadas)
- Transformación de tipos de datos:
  - Conversión de edades en texto a números
  - Limpieza de símbolos monetarios y conversión a float
  - Normalización de valores categóricos
- Corrección de valores negativos (distancias convertidas de millas a km)
- Homogeneización de texto (mayúsculas a minúsculas)
- Guardado del dataset limpio como `HR_RAW_v2.csv`

#### 🔧 PARTE 3: Gestión de Valores Nulos
**Archivo:** `PARTE-3_project-da-promo-52-modulo-3-team-4-NULOS.ipynb`

**Contenido:**
- Análisis detallado del porcentaje de valores nulos por columna
- Estrategias de imputación diferenciadas:
  - **Columnas categóricas**: Imputación con valores lógicos ("non-travel", "full time", "ns/nc")
  - **Columnas numéricas**: Uso de moda, mediana, KNN Imputer según el caso
- Traducción completa de valores al español para mejor interpretación
- Renombrado final de columnas para visualizaciones
- **Generación del dataset final `HR_RAW_v3.csv`** completamente procesado

**Técnicas aplicadas:**
- KNN Imputer para variables continuas correlacionadas
- Simple Imputer con estrategia de mediana
- Imputación manual basada en conocimiento del dominio

#### 📈 PARTE 4: Visualización de Datos
**Archivo:** `PARTE-4_project-da-promo-52-modulo-3-team-4-VISUALIZACION.ipynb`

**Contenido:**
- Definición de paleta de colores corporativos
- Prueba de hipótesis sobre las columnas numéricas
- **Gráfico 1**: Comparación de variables emocionales (satisfacción y compromiso)
- **Gráfico 2**: Análisis de historia laboral mediante boxplots
- **Gráfico 3**: Relación entre antigüedad y tiempo con el mismo responsable
- **Gráfico 4**: Análisis de conciliación laboral por tipo de jornada
- **Gráfico 5**: Comparación entre evaluación de desempeño e incremento salarial
- **Perfil del empleado desvinculado**: Análisis demográfico detallado
- BONUS: Inserción de datos en SQL 

**Principales insights:**
- Empleados desvinculados muestran menor satisfacción en todas las dimensiones
- Menor antigüedad y experiencia laboral en el grupo desvinculado
- Perfil típico: hombre de 33-34 años con menor satisfacción en conciliación

#### 📊 ETL - Carpeta con el código de ETL (incompleto)
**Archivos:** `main.py`
              `funciones.py`
              `graficos.py`
              `load_sql.py`
              `HR RAW DATA.csv`
              `HR LIMPIO.csv`

Contiene toda la información indicada en todos los puntos anteriores en formato python.

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib**: Visualización de datos
- **Seaborn**: Visualizaciones estadísticas avanzadas
- **Scikit-learn**: Algoritmos de imputación (KNNImputer, SimpleImputer)

## 📊 Dataset Final (HR_RAW_v3.csv)

El dataset final procesado contiene:
- **1,614 registros** de empleados
- **34 columnas** después de la limpieza
- **0 valores nulos** (totalmente imputado)
- **Datos completamente en español** para facilitar interpretación
- **Tipos de datos optimizados** para análisis

### Principales Variables
- Variables demográficas: edad, género, estado civil
- Variables laborales: departamento, puesto, antigüedad
- Variables de satisfacción: trabajo, relaciones, conciliación
- Variables de compensación: salario, incrementos, beneficios

## 🔍 Principales Hallazgos

1. **Factor Crítico**: La satisfacción laboral en múltiples dimensiones es significativamente menor en empleados desvinculados
2. **Perfil de Riesgo**: Empleados con menor antigüedad y menor satisfacción en conciliación laboral
3. **Reconocimiento**: El sistema de evaluación y compensación funciona correctamente
4. **Demografía**: Mayor proporción de hombres en el grupo desvinculado (62%)

## 🚀 Metodología

El proyecto sigue metodología Agile/Scrum con las siguientes fases:

1. **Exploración** → Comprensión inicial de los datos
2. **Limpieza** → Preparación y normalización
3. **Imputación** → Gestión de valores faltantes
4. **Visualización** → Análisis y presentación de insights
5. **Conclusiones** → Recomendaciones para la empresa

## 👥 Equipo

**Equipo 4 - Promoción 52 - Módulo 3**
- Proyecto desarrollado como parte del programa de Data Analytics

## 📝 Próximos Pasos

- Implementación de modelos predictivos de rotación
- Desarrollo de dashboard interactivo
- Creación de sistema ETL automatizado
- Diseño de base de datos optimizada para análisis de HR

## 🌈 Presentación

https://www.canva.com/design/DAGpmKBoF8w/sCUkzv78Kklt-e5wLq-gfw/edit?utm_content=DAGpmKBoF8w&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

---

*Este proyecto demuestra el poder del análisis de datos para la toma de decisiones estratégicas en Recursos Humanos, proporcionando insights accionables para mejorar la retención del talento.*