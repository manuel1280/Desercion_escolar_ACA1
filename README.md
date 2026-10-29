# Predicción de deserción estudiantil con ML clásico (ACA N.º 1)

Proyecto de la asignatura **Aprendizaje Automático** · Especialización en Inteligencia Artificial · CUN
Docente: Dr. Hamilton Rivera Flor · Pilar: Educación

Modelo de clasificación que estima, para cada estudiante, si terminará en **abandono (*Dropout*)**, seguirá **matriculado (*Enrolled*)** o se **graduará (*Graduate*)**, a partir de datos académicos, socioeconómicos y demográficos disponibles al matricularse y durante el primer año.

## Integrantes
- Johan Manuel Buitrago Segura
- Yadlif Mateo Hernández Zapata
- Cristian David Muñoz Sánchez
- Edna Catalina Osorio Alzate
- Camilo José Patiño Rodríguez

## Enlaces
- App (Streamlit): [Permanencia · ACA1](https://permanencia-aca1.streamlit.app/)
- Video de sustentación: [pendiente]
- Póster (PDF): [pendiente]

## Datos
- **Fuente:** UCI *Predict Students Dropout and Academic Success* (id 697), 4.424 estudiantes × 36 variables + objetivo `Target`.
- Sin nulos ni duplicados. Distribución del objetivo: *Graduate* ≈50 %, *Dropout* ≈32 %, *Enrolled* ≈18 %.
- Los códigos categóricos (carrera, ocupación, etc.) se tratan como **nominales** (one-hot), no como números. Los valores atípicos se auditan pero no se eliminan.

## Metodología
1. Limpieza y auditoría de calidad (nombres en snake_case, nulos, duplicados, ceros sospechosos, atípicos por IQR, cardinalidad).
2. Partición 80/20 estratificada (semilla 42) y preprocesamiento dentro de un `Pipeline` para evitar fuga de datos.
3. Comparación de modelos con validación cruzada estratificada de 5 folds y métrica principal **F1 macro**. Desbalance tratado con `class_weight="balanced"`.
4. Ajuste de hiperparámetros del mejor candidato (GridSearchCV) y evaluación **una sola vez** en el test.
5. Pruebas de robustez: etiquetas barajadas, otras particiones e intervalo de confianza por bootstrap.

## Resultados
**Comparación en validación cruzada (F1 macro)**

| Modelo | F1 macro (CV) |
|---|---|
| Gradient Boosting | 0,715 |
| SVM (RBF) | 0,714 |
| Regresión logística | 0,707 |
| Random Forest | 0,684 |
| Baseline (clase mayoritaria) | 0,222 |

Los tres primeros quedan dentro de una desviación estándar; esto no demuestra equivalencia estadística ni una ventaja concluyente. Se eligió Gradient Boosting por tener la CV más alta; la regresión logística es la alternativa más simple y explicable.

**Modelo final en el test (885 estudiantes)**

| Métrica | Valor |
|---|---|
| F1 macro | 0,694 (IC 95 % bootstrap: 0,664–0,725) |
| Accuracy | 0,729 |
| AUC (OvR) | 0,885 |
| *Dropout*: precisión / recall | 0,822 / 0,683 |
| *Enrolled*: F1 | 0,511 (clase más difícil) |

Variables más influyentes (importancia por permutación): unidades aprobadas del 2.º semestre, unidades aprobadas del 1.er semestre y matrícula al día.

**Robustez:** con etiquetas barajadas el F1 macro cae a 0,319 (cerca del azar); otras particiones (semillas 0–4) dieron entre 0,725 y 0,740, lo que sugiere que la partición oficial es conservadora.

### Versión preventiva (sin variables del 2.º semestre)
Con `USAR_2DO_SEM = False` el modelo usa solo información hasta el 1.er semestre.

| Métrica (test) | Con 2.º semestre | Preventiva |
|---|---|---|
| F1 macro | 0,694 | 0,667 |
| Recall *Dropout* | 0,683 | 0,669 |
| F1 *Enrolled* | 0,511 | 0,465 |
| AUC (OvR) | 0,885 | 0,858 |

La pérdida es modesta (≈0,03 de F1), pero del mismo orden que la incertidumbre del test: no debe leerse con precisión decimal. La app usa el artefacto oficial de ambos semestres y permite entrenar una variante experimental de primer semestre en el laboratorio.

## Estructura del repositorio
```
├── ACA1_Desercion_Limpieza_y_Modelo.ipynb   # limpieza, modelado, evaluación y artefactos
└── artefactos/
    ├── modelo_desercion.pkl        # pipeline completo (preprocesamiento + modelo)
    ├── model_card.json             # columnas de entrada, rangos, categorías y métricas
    ├── requirements_modelo.txt     # versiones de librerías necesarias para cargar el .pkl
    └── dataset_procesado.csv       # datos tras la limpieza
```

## Cómo reproducirlo
1. Abrir el notebook en Google Colab y ejecutar **Entorno de ejecución → Ejecutar todo**.
2. El notebook descarga el dataset oficial de la UCI. Si la descarga falla, subir `data.csv` al entorno de Colab.

## Cómo usar el modelo
El `.pkl` es un pipeline completo: recibe un `DataFrame` con las columnas de `columnas_entrada` (ver `model_card.json`) y devuelve probabilidades por clase. Instala antes las versiones de `requirements_modelo.txt`.

```python
import joblib, pandas as pd
modelo = joblib.load("artefactos/modelo_desercion.pkl")
fila = pd.DataFrame([{...}])           # una fila con las columnas de model_card.json
modelo.predict_proba(fila)             # columnas en el orden de modelo.classes_
```

> El `.pkl` depende de la versión de las librerías: con otras versiones, `joblib.load` puede fallar o dar resultados distintos.

## Limitaciones
- Con las variables del 2.º semestre el modelo detecta la deserción **tarde**; la versión preventiva pierde poco rendimiento.
- El modelo usa género, nacionalidad, estado civil y ocupación de los padres. Usarlo para decidir sobre estudiantes concretos exige una revisión de equidad.
- Los datos vienen de una sola institución; no se puede asumir que generalicen a otros contextos sin revalidar.
- Con F1 macro ≈0,69 y *Enrolled* en 0,51 sirve como **alerta temprana**, no para decisiones individuales.

## Referencia
Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *UCI Predict Students Dropout and Academic Success* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success


## App Streamlit

La app mantiene el pipeline oficial y ofrece cinco vistas:

- **Panorama:** métricas reproducidas del test, distribución del dataset, matriz de confusión, ROC y sensibilidad del umbral.
- **Simulador:** edición de las 36 variables, probabilidades por clase y descarga del escenario.
- **Predicción por lotes:** CSV con validación de columnas, valores y consistencia; descarga de resultados.
- **Laboratorio:** ajuste de tasa de aprendizaje, profundidad, hojas, L2 e iteraciones, con 5 folds sobre train. Opción de primer semestre. El modelo experimental puede usarse para predecir en la misma sesión.
- **Metodología:** decisiones, resultados y limitaciones del estudio.

### Ejecutar localmente (Python 3.13)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

### Validar

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

### Desplegar en Streamlit Community Cloud

Despliegue actual: `manuel1280/Desercion_escolar_ACA1`, rama `main`, archivo `app.py`.
La copia conserva el origen [Yadkomateo/Desercion_escolar_ACA1](https://github.com/Yadkomateo/Desercion_escolar_ACA1).
En **Advanced settings**, elegir **Python 3.13**. No necesita secrets ni servicios de pago.
El archivo raíz `requirements.txt` instala las versiones del artefacto guardado.
Guía: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

El experimento se guarda solo en la sesión; cambiar controles no lo reentrena hasta pulsar el botón.
La app no escribe los CSV cargados en disco. Usar archivos anónimos y sin identificadores.
La ficha y los datos públicos se incluyen para evitar dependencias de red en el arranque.

La revisión técnica y los hallazgos adicionales están en [ANALISIS_PROYECTO.md](ANALISIS_PROYECTO.md).
