import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from model_service import (CARD, FEATURES, SECOND_SEM, evaluate, load_data,
                           load_model, predict, split_data, train_experiment, validate_frame)


def test_saved_model_reproduces_official_test_metrics():
    x_train, x_test, y_train, y_test = split_data()
    assert len(x_train) == 3539 and len(x_test) == 885
    assert not set(x_train.index) & set(x_test.index)
    metrics, results = evaluate(load_model(), x_test, y_test)
    for key, value in metrics.items():
        assert round(value, 4) == CARD['metricas_test'][key]
    assert np.allclose(results.filter(like='p_').sum(axis=1), 1)


def test_threshold_changes_alerts_not_predictions():
    frame = load_data()[FEATURES].head(100)
    model = load_model()
    low, high = predict(model, frame, .2), predict(model, frame, .8)
    pd.testing.assert_series_equal(low.prediccion, high.prediccion)
    assert low.alerta_dropout.sum() > high.alerta_dropout.sum()


def test_uci_aliases_and_invalid_inputs():
    frame = load_data()[FEATURES].head(5)
    aliases = {v: k for k, v in CARD['mapa_columnas_original_a_snake'].items()}
    normalized, _ = validate_frame(frame.rename(columns=aliases))
    pd.testing.assert_frame_equal(normalized, frame)
    for field, value in [('gender', 2), ('age_at_enrollment', np.inf), ('admission_grade', 'bad'),
                         ('age_at_enrollment', 20.5), ('admission_grade', 201),
                         ('curricular_units_1st_sem_approved', 100)]:
        bad = frame.astype(object).copy()
        bad.iloc[0, bad.columns.get_loc(field)] = value
        with pytest.raises(ValueError):
            validate_frame(bad)
    with pytest.raises(ValueError, match='Faltan columnas'):
        validate_frame(frame.drop(columns='course'))
    unknown = frame.copy()
    unknown['course'] = 99999
    valid, warnings = validate_frame(unknown)
    assert warnings and len(predict(load_model(), valid)) == 5


def test_all_pages_and_simulator_react_without_errors():
    at = AppTest.from_file('app.py', default_timeout=45).run()
    assert not at.exception
    for page in ['Simulador', 'Predicción por lotes', 'Laboratorio', 'Metodología']:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    at.sidebar.radio[0].set_value('Simulador').run()
    at.selectbox(key='input_Ejemplo A_course').set_value(9500).run()
    assert not at.exception
    at.number_input(key='input_Ejemplo A_curricular_units_1st_sem_approved').set_value(26).run()
    assert len(at.error) == 1


def test_preventive_experiment_excludes_second_semester_and_official_test():
    experiment = train_experiment(dict(learning_rate=.1, max_depth=3, max_leaf_nodes=15,
                                      l2_regularization=1., max_iter=50), preventive=True)
    assert not set(SECOND_SEM) & set(experiment['features'])
    assert len(experiment['features']) == 30
    assert len(experiment['cv_f1']) == 5
    assert np.isfinite(experiment['cv_f1']).all()
    _, x_test, _, _ = split_data()
    assert len(predict(experiment['model'], x_test[experiment['features']].head(3))) == 3
    assert load_model().named_steps['clf'].max_iter == 100


def test_original_data_is_accepted_with_quality_warnings():
    valid, warnings = validate_frame(load_data())
    assert len(valid) == 4424
    assert any('sin evaluación' in w or 'without_evaluations' in w for w in warnings)


def test_train_then_select_experiment_in_ui():
    at = AppTest.from_file('app.py', default_timeout=45).run()
    at.sidebar.radio[0].set_value('Laboratorio').run()
    at.checkbox[0].check().run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.session_state.experiment['features']) == 30
    at.sidebar.radio[0].set_value('Simulador').run()
    at.selectbox(key='individual_model').set_value('Experimento de esta sesión').run()
    assert not at.exception
    assert len(at.number_input) == 13
    assert not at.error
