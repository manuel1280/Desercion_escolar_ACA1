# Revisión del proyecto y decisiones de la app

## Hallazgos verificados

- El artefacto es `HistGradientBoostingClassifier`, no `GradientBoostingClassifier`, envuelto en un pipeline con preprocesamiento.
- Los parámetros finales son learning_rate=0.1, max_depth=3, max_leaf_nodes=15 y l2_regularization=1.0; conserva max_iter=100, class_weight=balanced y random_state=42.
- La partición estratificada 80/20 con semilla 42 reproduce 3.539 filas train y 885 test. Al cargar el artefacto se reproducen F1 macro=0.6934869, accuracy=0.7288136, recall Dropout=0.6830986 y AUC OvR macro=0.8847452.
- El modelo oficial requiere 36 variables, incluidas seis del segundo semestre. No debe presentarse como predicción al momento del ingreso.
- La CV antes del ajuste es 0.7153 y después del GridSearch es 0.7307. La cifra posterior tiene optimismo de selección; no sustituye la métrica del test.
- Estar dentro de una desviación estándar no demuestra equivalencia estadística entre modelos.
- El control de etiquetas barajadas se realizó con regresión logística. No es una prueba directa de robustez del clasificador final.
- Las particiones alternativas se solapan y reutilizan hiperparámetros seleccionados previamente. No son validación externa.
- La ficha no documenta calibración independiente de probabilidades ni una revisión de equidad por subgrupos.
- Auditoría adicional del CSV: tres registros tienen unidades sin evaluación del segundo semestre superiores a las matriculadas (12/6, 7/6, 8/5). Se conservan, siguiendo la metodología de no borrar atípicos; la app avisa de esta inconsistencia al recibirla.

## Decisiones de implementación

- El modelo oficial se carga sin reentrenarlo ni modificar el pickle.
- El simulador permite editar todas las entradas y visualizar probabilidades por clase.
- El umbral de alerta de abandono es una regla adicional; no modifica la clase argmax ni las probabilidades. Se comunica claramente.
- El laboratorio clona la configuración del pipeline y vuelve a entrenar con los hiperparámetros elegidos. La imputación, escalado y one-hot se ajustan dentro de cada fold. Se utiliza exclusivamente el 80 % de entrenamiento original; el test no se consulta para seleccionar parámetros.
- La opción preventiva elimina las seis variables del segundo semestre antes de ajustar el preprocesamiento y el clasificador. Es un experimento nuevo; no se le atribuyen las métricas históricas de la variante preventiva del notebook.
- El modelo experimental queda en la sesión de cada visitante y se puede seleccionar para inferencia individual o por lotes. No se cambia el artefacto oficial compartido.
- Límite de CSV: 5 MB y 10.000 filas. Se validan columnas, números finitos, códigos binarios, enteros y consistencia académica. Las categorías desconocidas usan el manejo del pipeline y generan aviso.
- Los archivos cargados se procesan en memoria y no se escriben en disco. No se cargan pickles de visitantes. Los datos públicos y el diccionario UCI se incluyen en el repositorio: el arranque no necesita descargar el dataset.
- El cómputo del entrenamiento limita los hilos a dos y ejecuta la CV en serie para reducir uso de recursos en Community Cloud. Cada entrenamiento exige pulsar el botón; cambiar un control no dispara entrenamientos.

## Alcance de producción

La app está preparada para demostración académica en Streamlit Community Cloud. No es un sistema institucional validado de decisiones sobre estudiantes. Para una operación real se requieren validación externa, evaluación temporal, calibración, revisión de equidad y controles de acceso adecuados a los datos que se procesen.

## Verificación

Siete pruebas automatizadas pasan: reproducción de métricas del test, umbral independiente de la clase, validación de entradas y alias UCI, navegación por todas las vistas, entrenamiento preventivo, conservación de las filas originales con avisos y uso del experimento desde el simulador. `pip check` no detecta conflictos. La app se comprobó visualmente en Chrome y el despliegue cargó las métricas oficiales en https://permanencia-aca1.streamlit.app/. La CV con parámetros oficiales también se reproduce: F1 macro medio 0.7306785. Se fija PyArrow 24.0.0, la versión aplicada por Community Cloud al detectar un fallo conocido en 26.0.0.

## Fuentes

- Notebook, README y artefactos del repositorio original.
- [UCI, dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success): diccionario, procedencia y licencia CC BY 4.0.
- [Documentación de despliegue Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
