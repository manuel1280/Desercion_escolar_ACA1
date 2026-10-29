"""ACA1 · Observatorio de permanencia estudiantil."""
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve

from labels import CLASS_LABELS, COLORS, LABELS, option_label
from model_service import (CARD, CLASSES, FEATURES, INTEGER_FEATURES, SECOND_SEM,
                           evaluate, load_data, load_model, predict, split_data,
                           train_experiment, validate_frame)

st.set_page_config(page_title='Permanencia · ACA1', page_icon='🎓', layout='wide')
st.markdown('''<style>
.block-container {max-width:1400px; padding-top:2.2rem;}
[data-testid="stSidebar"] {border-right:1px solid #dae2ec;}
h1 {letter-spacing:-1.5px; font-weight:750!important;}
h2,h3 {letter-spacing:-.5px;}
[data-testid="stMetric"] {background:white; border:1px solid #e0e6ee; border-radius:14px; padding:16px;}
.hero {background:#132D46; color:white; padding:28px 32px; border-radius:18px; margin:0 0 24px;}
.hero p {color:#c5d9e8; max-width:850px; margin-bottom:0;}
.eyebrow {color:#76d6c5; font-size:12px; font-weight:700; letter-spacing:2px; margin-bottom:8px;}
.hero h1 {color:white; font-size:36px; padding:0 0 10px;}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def official_model():
    return load_model()


@st.cache_data
def dataset():
    return load_data()


@st.cache_data
def official_evaluation():
    _, x_test, _, y_test = split_data()
    metrics, results = evaluate(official_model(), x_test, y_test)
    return metrics, results, y_test


def plot(fig, key=None):
    fig.update_layout(font=dict(family='Arial', color='#16283F'), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key, config={'displayModeBar': False})


def probabilities(result):
    values = [float(result[f'p_{c}']) for c in CLASSES]
    fig = go.Figure(go.Bar(x=values, y=[CLASS_LABELS[c] for c in CLASSES], orientation='h',
                          marker_color=[COLORS[c] for c in CLASSES],
                          text=[f'{p:.1%}' for p in values], textposition='auto'))
    fig.update_layout(height=235, xaxis=dict(range=[0, 1], tickformat='.0%'), yaxis=dict(autorange='reversed'))
    plot(fig, 'individual_probabilities')


def model_choice(key):
    options = ['Modelo oficial']
    if 'experiment' in st.session_state:
        options.append('Experimento de esta sesión')
    choice = st.selectbox('Modelo para predecir', options, key=key)
    if choice == 'Modelo oficial':
        return official_model(), FEATURES, choice
    experiment = st.session_state.experiment
    st.info('Experimento activo: ' + ('solo primer semestre' if experiment['preventive'] else 'ambos semestres') + '. Sus métricas son de validación cruzada sobre train.')
    return experiment['model'], experiment['features'], choice


with st.sidebar:
    st.markdown('### 🎓 PERMANENCIA')
    st.caption('OBSERVATORIO ESTUDIANTIL · ACA1')
    page = st.radio('Explorar', ['Panorama', 'Simulador', 'Predicción por lotes', 'Laboratorio', 'Metodología'], label_visibility='collapsed')
    st.divider()
    st.markdown('**Modelo de referencia**')
    st.write('HistGradientBoosting')
    st.caption('36 variables · 3 clases · Semilla 42')
    st.markdown('**Momento de predicción**')
    st.caption('Al finalizar el segundo semestre. La variante de primer semestre se entrena en el laboratorio.')
    st.divider()
    st.caption('CUN · Especialización en IA\n\nAprendizaje Automático')

st.markdown('''<div class="hero"><div class="eyebrow">EDUCACIÓN + MACHINE LEARNING</div>
<h1>Entender el riesgo. Explorar la permanencia.</h1>
<p>Un laboratorio para visualizar resultados académicos, simular perfiles y comprender el modelo de deserción estudiantil del proyecto ACA1.</p></div>''', unsafe_allow_html=True)

if page == 'Panorama':
    st.subheader('El modelo, de un vistazo')
    st.caption('Evaluación del pipeline guardado sobre la partición oficial de prueba: 885 registros, 20 % del conjunto.')
    metrics, results, y_test = official_evaluation()
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [('F1 macro', 'f1_macro'), ('Exactitud', 'accuracy'), ('Recall de abandono', 'recall_dropout'), ('AUC · uno contra el resto', 'roc_auc_ovr_macro')]):
        col.metric(label, f'{metrics[key]:.3f}')
    st.caption('F1 macro: IC 95 % bootstrap reportado en el notebook = 0,664–0,725. Las probabilidades no tienen una calibración independiente documentada.')
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown('#### ¿Qué hay en los datos?')
        counts = dataset().target.value_counts().reindex(CLASSES)
        fig = go.Figure(go.Pie(labels=[CLASS_LABELS[c] for c in CLASSES], values=counts, hole=.64,
                               marker_colors=[COLORS[c] for c in CLASSES], textinfo='percent+label'))
        fig.update_layout(height=330, showlegend=False, annotations=[dict(text='4.424<br>estudiantes', x=.5, y=.5, showarrow=False, font_size=20)])
        plot(fig)
        st.caption('Datos públicos UCI · una institución de educación superior de Portugal. No representan una muestra de estudiantes de Colombia.')
    with b:
        st.markdown('#### ¿Dónde acierta y dónde se confunde?')
        matrix = confusion_matrix(y_test, results.prediccion, labels=CLASSES)
        fig = px.imshow(matrix, x=[CLASS_LABELS[c] for c in CLASSES], y=[CLASS_LABELS[c] for c in CLASSES],
                         text_auto=True, color_continuous_scale='Teal', labels={'x': 'Predicción', 'y': 'Resultado real', 'color': 'Estudiantes'})
        fig.update_layout(height=330, coloraxis_showscale=False)
        plot(fig)
        st.caption('Conteos reales del test. «Matriculado» es la clase más difícil de distinguir.')
    with st.expander('Ver métricas por clase y curvas ROC'):
        report = classification_report(y_test, results.prediccion, output_dict=True)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
        fig = go.Figure()
        for c in CLASSES:
            fpr, tpr, _ = roc_curve(y_test == c, results[f'p_{c}'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=CLASS_LABELS[c], line_color=COLORS[c]))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Azar', line=dict(dash='dash', color='#98A6B7')))
        fig.update_layout(xaxis_title='Tasa de falsos positivos', yaxis_title='Sensibilidad', height=350)
        plot(fig)
    st.markdown('#### Sensibilidad de las alertas de abandono')
    threshold = st.slider('Umbral de alerta', .05, .95, .50, .05, help='Regla adicional: alerta si P(Abandono) ≥ umbral. No modifica la clase más probable ni reentrena el modelo.')
    alert = results.p_Dropout >= threshold
    true = y_test == 'Dropout'
    a, b, c = st.columns(3)
    a.metric('Alertas en el test', int(alert.sum()))
    b.metric('Precisión de la alerta', f'{precision_score(true, alert, zero_division=0):.1%}')
    c.metric('Abandonos detectados', f'{recall_score(true, alert):.1%}')
    st.caption('Exploración descriptiva del test ya publicado; no constituye una nueva evaluación independiente ni un umbral clínico o institucional validado.')

elif page == 'Simulador':
    st.subheader('Simula un perfil de estudiante')
    model, features, model_name = model_choice('individual_model')
    st.caption('Los campos parten de un registro público de ejemplo. Cambia sus valores para explorar escenarios; las variaciones no demuestran causalidad.')
    source = st.selectbox('Perfil de partida', ['Ejemplo A', 'Ejemplo B', 'Ejemplo C'])
    idx = ['Ejemplo A', 'Ejemplo B', 'Ejemplo C'].index(source)
    seed = dataset().iloc[[0, 2, 5][idx]]
    inputs, output = st.columns([1.45, 1], gap='large')
    values = {}
    with inputs:
        groups = [('Trayectoria académica', [c for c in features if 'curricular_units' in c]),
                  ('Ingreso y programa', [c for c in features if c in ['course', 'application_mode', 'application_order', 'previous_qualification', 'previous_qualification_grade', 'admission_grade', 'daytime_evening_attendance', 'age_at_enrollment']]),
                  ('Contexto económico y personal', [c for c in features if 'curricular_units' not in c and c not in ['course', 'application_mode', 'application_order', 'previous_qualification', 'previous_qualification_grade', 'admission_grade', 'daytime_evening_attendance', 'age_at_enrollment']])]
        for g, columns in groups:
            with st.expander(g, expanded=g == 'Trayectoria académica'):
                pair = st.columns(2)
                for n, field in enumerate(columns):
                    with pair[n % 2]:
                        key = f'input_{source}_{field}'
                        if field in CARD['nominales'] or field in CARD['binarias']:
                            options = CARD['nominales'].get(field, [0, 1])
                            values[field] = st.selectbox(LABELS[field], options, index=options.index(int(seed[field])),
                                                        format_func=lambda v, f=field: option_label(f, v), key=key, help=field)
                        else:
                            integer = field in INTEGER_FEATURES
                            low = -10.0 if field in ('gdp', 'inflation_rate') else 0.0
                            high = max(CARD['numericas'][field]['max'], 20 if field.endswith('_sem_grade') else 200 if field in ('admission_grade', 'previous_qualification_grade') else CARD['numericas'][field]['max'])
                            values[field] = st.number_input(LABELS[field], min_value=int(low) if integer else low,
                                                           max_value=int(high) if integer else float(high),
                                                           value=int(seed[field]) if integer else float(seed[field]),
                                                           step=1 if integer else .1, key=key, help=field)
    with output:
        with st.container(border=True):
            st.markdown('#### Resultado del escenario')
            st.caption(model_name + (' · 1.er semestre' if len(features) < 36 else ' · Ambos semestres'))
            threshold = st.slider('Umbral de alerta de abandono', .05, .95, .50, .05)
            try:
                frame, warnings = validate_frame(pd.DataFrame([values]), features)
                result = predict(model, frame, threshold).iloc[0]
                st.metric('Resultado más probable', CLASS_LABELS[result.prediccion])
                probabilities(result)
                if result.alerta_dropout:
                    st.warning(f'Alerta activada: P(Abandono) = {result.p_Dropout:.1%} ≥ {threshold:.0%}.')
                else:
                    st.info(f'Sin alerta con este umbral: P(Abandono) = {result.p_Dropout:.1%} < {threshold:.0%}.')
                for warning in warnings:
                    st.warning(warning)
                st.caption('La alerta y la clase más probable son reglas distintas. El umbral no cambia las probabilidades.')
                export = frame.join(predict(model, frame, threshold))
                export['modelo'] = model_name
                export['umbral_alerta'] = threshold
                st.download_button('Descargar escenario y predicción', export.to_csv(index=False).encode('utf-8-sig'), 'escenario.csv', 'text/csv')
            except ValueError as exc:
                st.error(str(exc))
        st.caption('Uso académico y exploratorio. Los resultados requieren interpretación humana y no determinan acceso a educación, becas o sanciones.')

elif page == 'Predicción por lotes':
    st.subheader('De un perfil a un conjunto de escenarios')
    model, features, model_name = model_choice('batch_model')
    st.write('Carga un CSV con las columnas del modelo y obtén la clase más probable y las tres probabilidades por fila.')
    st.caption('Usa datos anónimos, sin nombres ni documentos. El archivo se procesa en la memoria de tu sesión y esta app no lo guarda en disco.')
    st.download_button('Descargar CSV de ejemplo', dataset()[features].head(5).to_csv(index=False).encode('utf-8-sig'), 'plantilla_estudiantes.csv', 'text/csv')
    upload = st.file_uploader('Archivo CSV · máximo 5 MB / 10.000 filas', type=['csv'])
    threshold = st.slider('Umbral de alerta para el lote', .05, .95, .50, .05)
    if upload:
        try:
            raw = pd.read_csv(io.BytesIO(upload.getvalue()), sep=None, engine='python', nrows=10001)
            frame, warnings = validate_frame(raw, features)
            for warning in warnings:
                st.warning(warning)
            result = predict(model, frame, threshold)
            a, b, c = st.columns(3)
            a.metric('Perfiles procesados', len(result))
            b.metric('Alertas de abandono', int(result.alerta_dropout.sum()))
            c.metric('Probabilidad media de abandono', f'{result.p_Dropout.mean():.1%}')
            plot(px.histogram(result, x='p_Dropout', nbins=20, labels={'p_Dropout': 'Probabilidad de abandono'}, color_discrete_sequence=['#137C72']))
            st.dataframe(result, use_container_width=True)
            export = frame.join(result)
            export['modelo'] = model_name
            export['umbral_alerta'] = threshold
            st.download_button('Descargar predicciones', export.to_csv(index=False).encode('utf-8-sig'), 'predicciones.csv', 'text/csv')
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f'No se pudo procesar el archivo: {exc}')
    with st.expander('Columnas requeridas y formato'):
        st.dataframe(pd.DataFrame({'Columna': features, 'Descripción': [LABELS[c] for c in features]}), hide_index=True, use_container_width=True)
        st.caption('Se aceptan nombres snake_case y los originales UCI; separadores coma o punto y coma; decimales con punto. Las columnas extra no se usan. No se admiten celdas vacías, infinitos, códigos binarios inválidos ni conteos fraccionarios.')

elif page == 'Laboratorio':
    st.subheader('Ajusta el modelo y observa qué cambia')
    st.write('Cada experimento reentrena HistGradientBoosting y ejecuta 5 folds estratificados sobre los 3.539 registros de entrenamiento.')
    st.info('La partición de prueba oficial se mantiene fuera del ajuste. El experimento queda disponible en el simulador y en los lotes durante esta sesión.')
    with st.form('hyperparameters'):
        a, b, c = st.columns(3)
        with a:
            learning_rate = st.select_slider('Tasa de aprendizaje', options=[.03, .05, .1, .2], value=.1)
            max_depth = st.selectbox('Profundidad máxima', [3, 6, None], format_func=lambda v: 'Sin límite' if v is None else str(v))
        with b:
            max_leaf_nodes = st.selectbox('Máximo de hojas', [15, 31])
            l2_regularization = st.select_slider('Regularización L2', options=[0., .5, 1., 2., 5.], value=1.)
        with c:
            max_iter = st.select_slider('Iteraciones de boosting', options=[50, 100, 150, 200], value=100)
            preventive = st.checkbox('Usar solo el primer semestre', help='Elimina las seis variables del segundo semestre y reentrena. No sustituye el artefacto oficial ni replica necesariamente el ajuste preventivo del notebook.')
        st.caption('Se conservan class_weight="balanced", semilla 42, one-hot con frecuencia mínima 1 %, imputación y escalado dentro de cada fold.')
        run = st.form_submit_button('Entrenar y validar experimento', type='primary')
    if run:
        params = dict(learning_rate=learning_rate, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
                      l2_regularization=l2_regularization, max_iter=max_iter)
        with st.spinner('Entrenando y evaluando los 5 folds…'):
            st.session_state.experiment = train_experiment(params, preventive)
        st.success('Experimento listo. Ya puedes seleccionarlo para predecir.')
    if 'experiment' in st.session_state:
        exp = st.session_state.experiment
        st.markdown('#### Resultado del último experimento entrenado')
        st.caption('Cambiar los controles no actualiza este resultado hasta pulsar Entrenar.')
        a, b, c = st.columns(3)
        a.metric('F1 macro · media CV', f"{exp['cv_f1'].mean():.3f}")
        b.metric('Desviación entre folds', f"{exp['cv_f1'].std():.3f}")
        c.metric('Variables utilizadas', len(exp['features']))
        scores = pd.DataFrame({'Fold': list(range(1, 6)), 'F1 macro': exp['cv_f1'], 'Exactitud': exp['cv_accuracy']})
        plot(px.bar(scores, x='Fold', y='F1 macro', color_discrete_sequence=['#137C72'], range_y=[0, 1]))
        st.caption(f"Referencia histórica: F1 macro CV oficial tras GridSearch = {CARD['metricas_cv_f1_macro']:.4f}. Seleccionar configuraciones repetidamente sobre estos folds introduce optimismo; no es rendimiento en datos nuevos.")
        st.json({'primer_semestre': exp['preventive'], **exp['params']}, expanded=False)
        st.download_button('Descargar informe del experimento', json.dumps({'params': exp['params'], 'primer_semestre': exp['preventive'], 'features': exp['features'], 'seed': 42, 'cv_f1_macro': exp['cv_f1'].tolist(), 'cv_accuracy': exp['cv_accuracy'].tolist()}, indent=2), 'experimento.json', 'application/json')
        if st.button('Descartar experimento de la sesión'):
            del st.session_state.experiment
            st.rerun()

else:
    st.subheader('Del dato a la predicción')
    st.markdown('''
1. **Auditar y limpiar.** 4.424 registros, 36 variables, nombres normalizados; sin nulos ni duplicados reportados. Los atípicos se auditan y se conservan.
2. **Separar antes de aprender.** 80 % entrenamiento y 20 % prueba, estratificados, semilla 42.
3. **Preprocesar dentro del pipeline.** Categorías nominales con one-hot y agrupación de frecuencias menores al 1 %; imputación y escalado numérico. Cada fold ajusta su propio preprocesamiento.
4. **Comparar y ajustar.** Cinco folds estratificados, F1 macro y pesos de clase balanceados. GridSearchCV selecciona los hiperparámetros de HistGradientBoosting.
5. **Evaluar.** El notebook reporta la prueba final, robustez por particiones, etiquetas barajadas e intervalo bootstrap. La app reproduce las predicciones del artefacto oficial.
''')
    st.markdown('#### Selección del modelo · resultados del notebook')
    comparison = pd.DataFrame({'Modelo': ['Gradient Boosting', 'SVM (RBF)', 'Regresión logística', 'Random Forest', 'Clase mayoritaria'],
                               'F1 macro CV antes del ajuste': [.7153, .714, .7072, .684, .222]})
    plot(px.bar(comparison, x='F1 macro CV antes del ajuste', y='Modelo', orientation='h', color_discrete_sequence=['#137C72']))
    st.caption('Los tres primeros quedan dentro de una desviación estándar; esto no demuestra equivalencia estadística. La ventaja de Gradient Boosting no es concluyente.')
    st.markdown('#### Lo que el modelo puede y no puede decir')
    st.write('El artefacto usa información del segundo semestre: su capacidad para anticipar el abandono al ingreso es limitada. La clase «Matriculado» tiene F1 ≈ 0,51. Género, nacionalidad y variables familiares requieren evaluación de equidad antes de cualquier uso real. No hay validación externa ni calibración independiente de probabilidades documentada.')
    st.write('La variante preventiva del notebook reporta F1 macro ≈ 0,667 frente a 0,694 con ambos semestres. La variante del laboratorio es un nuevo experimento con los parámetros elegidos y muestra sus propias métricas de CV.')
    st.write('El control de etiquetas barajadas del notebook utiliza regresión logística, no HistGradientBoosting. Las pruebas con otras particiones reutilizan hiperparámetros y tienen solapamiento de datos: son análisis de sensibilidad, no validación externa.')
    st.markdown('#### Parámetros del artefacto oficial')
    st.json(CARD['mejores_parametros'])
    st.markdown('#### Fuente y equipo')
    st.markdown('[UCI · Predict Students’ Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) · [DOI](https://doi.org/10.24432/C5MC89) · CC BY 4.0. Realinho, Vieira Martins, Machado y Baptista (2021).')
    st.caption('Los nombres de categorías provienen del diccionario oficial UCI. Programa y estado civil se traducen al español; las demás categorías nominales conservan la etiqueta oficial en inglés para evitar alterar su significado.')
    st.write('Johan Manuel Buitrago Segura · Yadlif Mateo Hernández Zapata · Cristian David Muñoz Sánchez · Edna Catalina Osorio Alzate · Camilo José Patiño Rodríguez')
    st.caption('Docente: Dr. Hamilton Rivera Flor · CUN · Especialización en Inteligencia Artificial')

st.divider()
st.caption('ACA1 · Herramienta académica de exploración. Una predicción es una estimación, no una decisión sobre un estudiante.')
