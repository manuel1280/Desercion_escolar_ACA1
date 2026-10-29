"""Etiquetas de presentación; los valores enviados al modelo siguen siendo códigos UCI."""
import json
from model_service import ROOT

CLASS_LABELS = {'Dropout': 'Abandono', 'Enrolled': 'Matriculado', 'Graduate': 'Graduado'}
COLORS = {'Dropout': '#D66A50', 'Enrolled': '#D6A23D', 'Graduate': '#16877B'}
LABELS = {
    'marital_status': 'Estado civil', 'application_mode': 'Vía de admisión',
    'course': 'Programa académico', 'previous_qualification': 'Formación previa',
    'nationality': 'Nacionalidad', 'mothers_qualification': 'Formación de la madre',
    'fathers_qualification': 'Formación del padre', 'mothers_occupation': 'Ocupación de la madre',
    'fathers_occupation': 'Ocupación del padre', 'daytime_evening_attendance': 'Jornada',
    'displaced': 'Residencia desplazada (UCI)', 'educational_special_needs': 'Necesidades educativas especiales',
    'debtor': 'Tiene deuda', 'tuition_fees_up_to_date': 'Matrícula al día', 'gender': 'Género (registro UCI)',
    'scholarship_holder': 'Tiene beca', 'international': 'Estudiante internacional',
    'application_order': 'Orden de preferencia (0–9)',
    'previous_qualification_grade': 'Nota de formación previa (0–200)',
    'admission_grade': 'Nota de admisión (0–200)', 'age_at_enrollment': 'Edad al ingresar',
    'unemployment_rate': 'Tasa de desempleo (%)', 'inflation_rate': 'Inflación (%)', 'gdp': 'PIB (indicador UCI)',
}
for sem, label in [('1st', '1.er'), ('2nd', '2.º')]:
    for suffix, text in {'credited': 'Unidades reconocidas', 'enrolled': 'Unidades matriculadas',
                         'evaluations': 'Evaluaciones realizadas', 'approved': 'Unidades aprobadas',
                         'grade': 'Nota promedio (0–20)', 'without_evaluations': 'Unidades sin evaluación'}.items():
        LABELS[f'curricular_units_{sem}_sem_{suffix}'] = f'{text} · {label} semestre'
CATEGORIES = json.loads((ROOT / 'artefactos/diccionario_uci.json').read_text())['categories']
CATEGORIES['marital_status'] = dict(zip(map(str, range(1, 7)), ['Soltero/a', 'Casado/a', 'Viudo/a', 'Divorciado/a', 'Unión de hecho', 'Separado/a legalmente']))
CATEGORIES['course'] = dict(zip(map(str, [33,171,8014,9003,9070,9085,9119,9130,9147,9238,9254,9500,9556,9670,9773,9853,9991]),
    ['Tecnologías de biocombustibles', 'Animación y diseño multimedia', 'Trabajo social (nocturno)', 'Agronomía',
     'Diseño de comunicación', 'Enfermería veterinaria', 'Ingeniería informática', 'Equinocultura', 'Administración',
     'Trabajo social', 'Turismo', 'Enfermería', 'Higiene oral', 'Publicidad y marketing', 'Periodismo y comunicación',
     'Educación básica', 'Administración (nocturno)']))


def option_label(col, value):
    if col == 'gender':
        label = {0: 'Femenino', 1: 'Masculino'}[value]
    elif col == 'daytime_evening_attendance':
        label = {0: 'Nocturna', 1: 'Diurna'}[value]
    elif col in CATEGORIES:
        label = CATEGORIES[col].get(str(value), 'Código sin etiqueta')
    else:
        label = {0: 'No', 1: 'Sí'}[value]
    return f'{label} · {value}'
