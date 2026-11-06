"""Section/speaker/negation checks for the small Spanish/Chinese focus layer."""

import importlib.util
import unittest

from coder.clinical_query import diagnosis_focus


class MultilingualFocusTests(unittest.TestCase):
    def test_spanish_heading_stops_before_plan(self):
        text = 'Síntomas y antecedentes. Impresión diagnóstica: Bronquitis aguda. Plan: reposo.'
        self.assertEqual(diagnosis_focus(text), 'Bronquitis aguda.')
        self.assertEqual(diagnosis_focus('Diagnostico: Bronquitis aguda. Tratamiento: reposo.'),
                         'Bronquitis aguda.')

    def test_chinese_heading_and_full_width_punctuation(self):
        text = '主诉：咳嗽三天。初步诊断：急性支气管炎。计划：随访。'
        self.assertEqual(diagnosis_focus(text), '急性支气管炎。')

    def test_patient_suspicion_does_not_override_clinician(self):
        es = ('Paciente: Diagnóstico: neumonía. Creo que la tengo.\n'
              'Médico: No es neumonía. Es bronquitis aguda. Descanse.')
        zh = '患者：诊断：肺炎。我觉得是肺炎。\n医生：不是肺炎。这是急性支气管炎。请休息。'
        self.assertEqual(diagnosis_focus(es), 'bronquitis aguda.')
        self.assertEqual(diagnosis_focus(zh), '急性支气管炎。')

    def test_qualifying_negation_is_retained(self):
        self.assertEqual(diagnosis_focus('Valoración: Malaria no complicada.'),
                         'Malaria no complicada.')
        self.assertEqual(diagnosis_focus('评估：无并发症的疟疾。'), '无并发症的疟疾。')
        self.assertEqual(diagnosis_focus('医生：这是水痘，目前看没有并发症。'), '水痘，目前看没有并发症。')

    def test_negated_or_historical_diagnosis_is_not_extracted_as_current(self):
        for text in ('Diagnóstico: no neumonía.', 'Diagnóstico: se descarta neumonía.',
                     '诊断：排除肺炎。', '既往史：诊断：肺炎。', '既往诊断：肺炎。',
                     'Antecedentes: Diagnóstico: neumonía.'):
            with self.subTest(text=text):
                self.assertEqual(diagnosis_focus(text), text)

    def test_questions_and_hypothetical_statements_are_not_assessments(self):
        for text in ('Médico: ¿Es neumonía?', '医生：这是肺炎吗？',
                     'Médico: Es posible que se contagie.', 'Dolor y tos sin diagnóstico explícito.'):
            with self.subTest(text=text):
                self.assertEqual(diagnosis_focus(text), text)

    def test_abbreviations_do_not_split_lowercase_species_name(self):
        self.assertEqual(diagnosis_focus('Diagnóstico: Malaria por P. falciparum. Plan: tratamiento.'),
                         'Malaria por P. falciparum.')

    def test_no_missing_qualifier_is_invented(self):
        self.assertEqual(diagnosis_focus('诊断：新发现的房颤。'), '新发现的房颤。')
        self.assertEqual(diagnosis_focus('Diagnóstico: FA de nuevo diagnóstico.'), 'FA de nuevo diagnóstico.')

    def test_unlabelled_clinician_diagnosis_stops_before_advice(self):
        self.assertEqual(diagnosis_focus('El paciente tiene tos. El médico diagnosticó neumonía y recomendó tratamiento.'),
                         'neumonía')
        self.assertEqual(diagnosis_focus('患者咳嗽三天。医生诊断为肺炎，建议服药。'), '肺炎')

    @unittest.skipUnless(importlib.util.find_spec('abbreviations'), 'install requirements for abbreviation resolution')
    def test_only_explicit_note_local_abbreviations_are_expanded(self):
        text = 'ECG: fibrilación auricular (FA).\nDiagnóstico: FA de nuevo diagnóstico.'
        self.assertEqual(diagnosis_focus(text), 'fibrilación auricular de nuevo diagnóstico.')
        self.assertEqual(diagnosis_focus('Diagnóstico: FA de nuevo diagnóstico.'), 'FA de nuevo diagnóstico.')


if __name__ == '__main__':
    unittest.main()
