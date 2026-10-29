"""Inferencia y experimentación reproducibles sobre los artefactos de ACA1."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CARD = json.loads((ROOT / 'artefactos/model_card.json').read_text())
CLASSES = CARD['clases']
FEATURES = CARD['columnas_entrada']
SECOND_SEM = [c for c in FEATURES if '2nd_sem' in c]
INTEGER_FEATURES = set(CARD['nominales']) | set(CARD['binarias']) | {
    c for c in CARD['numericas'] if (c.startswith('curricular_units') and not c.endswith('_grade'))
} | {'age_at_enrollment', 'application_order'}


def load_model():
    # Solo se carga el artefacto del repositorio, nunca archivos pickle del usuario.
    model = joblib.load(ROOT / 'artefactos/modelo_desercion.pkl')
    if list(model.classes_) != CLASSES:
        raise ValueError('Las clases del modelo no coinciden con la ficha técnica.')
    return model


def load_data():
    return pd.read_csv(ROOT / 'artefactos/dataset_procesado.csv')


def split_data():
    data = load_data()
    return train_test_split(data[FEATURES], data.target, test_size=.2,
                            stratify=data.target, random_state=CARD['semilla'])


def validate_frame(frame, features=FEATURES):
    """Normaliza nombres UCI, valida entradas y devuelve avisos de extrapolación."""
    if frame.empty:
        raise ValueError('El CSV no contiene filas.')
    if len(frame) > 10000:
        raise ValueError('El límite es de 10.000 filas por archivo.')
    aliases = {k.casefold(): v for k, v in CARD['mapa_columnas_original_a_snake'].items()}
    df = frame.rename(columns=lambda c: aliases.get(str(c).strip().casefold(), str(c).strip()))
    if df.columns.duplicated().any():
        raise ValueError('Hay nombres de columnas duplicados después de normalizarlos.')
    missing = [c for c in features if c not in df]
    if missing:
        raise ValueError('Faltan columnas: ' + ', '.join(missing))
    df = df[features].copy()
    warnings = []
    for col in features:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except (ValueError, TypeError):
            raise ValueError(f'{col}: usa valores numéricos o códigos UCI.') from None
        if df[col].isna().any() or not np.isfinite(df[col]).all():
            raise ValueError(f'{col}: hay celdas vacías o valores no finitos.')
        if col in INTEGER_FEATURES and (df[col] % 1 != 0).any():
            raise ValueError(f'{col}: requiere números enteros.')
        if col in CARD['binarias'] and not df[col].isin([0, 1]).all():
            raise ValueError(f'{col}: solo admite 0 y 1.')
        if col in CARD['nominales'] and not df[col].isin(CARD['nominales'][col]).all():
            warnings.append(f'{col}: categoría no observada; el pipeline aplicará su manejo de categorías desconocidas.')
        if col in CARD['numericas']:
            limits = CARD['numericas'][col]
            if ((df[col] < limits['min']) | (df[col] > limits['max'])).any():
                warnings.append(f"{col}: fuera del rango observado [{limits['min']:g}, {limits['max']:g}].")
            if col not in ('gdp', 'inflation_rate') and (df[col] < 0).any():
                raise ValueError(f'{col}: no admite valores negativos.')
            ceiling = 200 if col in ('admission_grade', 'previous_qualification_grade') else 20 if col.endswith('_sem_grade') else None
            if ceiling is not None and (df[col] > ceiling).any():
                raise ValueError(f'{col}: la escala máxima es {ceiling}.')
    for sem in ('1st', '2nd'):
        enrolled = f'curricular_units_{sem}_sem_enrolled'
        for suffix in ('approved', 'without_evaluations'):
            other = f'curricular_units_{sem}_sem_{suffix}'
            if other in df and enrolled in df and (df[other] > df[enrolled]).any():
                if suffix == 'approved':
                    raise ValueError(f'{other}: no puede superar las unidades matriculadas.')
                warnings.append(f'{other}: supera las unidades matriculadas; revisa el registro. Esta inconsistencia también aparece en tres filas del dataset original.')
    return df, warnings


def predict(model, frame, threshold=.5):
    with threadpool_limits(limits=2):
        proba = model.predict_proba(frame)
    result = pd.DataFrame(proba, columns=[f'p_{c}' for c in model.classes_], index=frame.index)
    result['prediccion'] = np.asarray(model.classes_)[proba.argmax(axis=1)]
    result['alerta_dropout'] = result.p_Dropout >= threshold
    return result


def evaluate(model, x, y):
    results = predict(model, x)
    return {
        'f1_macro': f1_score(y, results.prediccion, average='macro'),
        'accuracy': accuracy_score(y, results.prediccion),
        'recall_dropout': recall_score(y, results.prediccion, labels=['Dropout'], average=None)[0],
        'roc_auc_ovr_macro': roc_auc_score(y, results[[f'p_{c}' for c in model.classes_]],
                                          multi_class='ovr', average='macro', labels=model.classes_),
    }, results


def train_experiment(params, preventive=False):
    """La CV y el ajuste final solo usan el 80 % train; el test oficial queda aparte."""
    x_train, _, y_train, _ = split_data()
    features = [c for c in FEATURES if not preventive or c not in SECOND_SEM]
    pipeline = clone(load_model())
    prep = pipeline.named_steps['prep']
    prep.transformers = [(name, transformer, [c for c in cols if c in features])
                         for name, transformer, cols in prep.transformers]
    pipeline.set_params(**{f'clf__{k}': v for k, v in params.items()})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CARD['semilla'])
    with threadpool_limits(limits=2):
        scores = cross_validate(pipeline, x_train[features], y_train, cv=cv,
                                scoring={'f1_macro': 'f1_macro', 'accuracy': 'accuracy'},
                                n_jobs=1, error_score='raise')
        pipeline.fit(x_train[features], y_train)
    return {'model': pipeline, 'features': features, 'params': params,
            'preventive': preventive, 'cv_f1': scores['test_f1_macro'],
            'cv_accuracy': scores['test_accuracy']}
