"""
This module performs unit tests for ms_marco_eval.py .

Command line:
/ms_marco_metrics$ PYTHONPATH=./bleu python ms_marco_eval_test.py

Creation Date : Dec-15-2016
Last Modified : Fri 16 December 2016 07:00:00 PT
Authors : Tri Nguyen <trnguye@microsoft.com>, Xia Song <xiaso@microsoft.com>, Tong Wang <tongw@microsoft.com>
"""

import os
import unittest

from ms_marco_eval import compute_metrics_from_files

TEST_DATA_FOLDER = 'sample_test_data'
MAX_BLEU_ORDER = 4

def generate_directory(test_data_file):
    """Generate the full path to test data file."""

    script_directory = os.path.dirname(__file__)
    test_file_directory = os.path.join(script_directory, TEST_DATA_FOLDER)
    test_file_directory = os.path.join(test_file_directory, test_data_file)
    return test_file_directory

class Test(unittest.TestCase):
    """Unit tests for ms_marco_eval.py ."""

    def test_same_answer(self):
        """Unit test for references and candidates sharing same answers."""

        reference_file = 'same_answer_test_references.json'
        candidate_file = 'same_answer_test_candidates.json'

        scores = compute_metrics_from_files(generate_directory(reference_file),
                                            generate_directory(candidate_file),
                                            MAX_BLEU_ORDER)
        self.assertEqual("%.5f" % scores['bleu_1'], '1.00000')
        self.assertEqual("%.5f" % scores['bleu_2'], '1.00000')
        self.assertEqual("%.5f" % scores['bleu_3'], '1.00000')
        self.assertEqual("%.5f" % scores['bleu_4'], '1.00000')
        self.assertEqual("%.5f" % scores['rouge_l'], '1.00000')

    def test_sample(self):
        """Unit test for sample references and candidates."""

        reference_file = 'sample_references.json'
        candidate_file = 'sample_candidates.json'

        scores = compute_metrics_from_files(generate_directory(reference_file),
                                            generate_directory(candidate_file),
                                            MAX_BLEU_ORDER)
        self.assertEqual("%.5f" % scores['bleu_1'], '0.00852')
        self.assertEqual("%.5f" % scores['bleu_2'], '0.00000')
        self.assertEqual("%.5f" % scores['bleu_3'], '0.00000')
        self.assertEqual("%.5f" % scores['bleu_4'], '0.00000')
        self.assertEqual("%.5f" % scores['rouge_l'], '0.03093')

    def test_no_answer(self):
        """Unit test for no-answer query."""

        reference_file = 'no_answer_test_references.json'
        candidate_file = 'no_answer_test_candidates.json'

        scores = compute_metrics_from_files(generate_directory(reference_file),
                                            generate_directory(candidate_file),
                                            MAX_BLEU_ORDER)
        self.assertEqual("%.5f" % scores['bleu_1'], '0.00000')
        self.assertEqual("%.5f" % scores['bleu_2'], '0.00000')
        self.assertEqual("%.5f" % scores['bleu_3'], '0.00000')
        self.assertEqual("%.5f" % scores['bleu_4'], '0.00000')
        self.assertEqual("%.5f" % scores['rouge_l'], '0.00000')

    def test_dev_set(self):
        """Unit test for dev set."""

        reference_file = 'dev_as_references.json'
        candidate_file = 'dev_first_sentence_as_candidates.json'

        scores = compute_metrics_from_files(generate_directory(reference_file),
                                            generate_directory(candidate_file),
                                            MAX_BLEU_ORDER)
        self.assertEqual("%.5f" % scores['bleu_1'], '0.17634')
        self.assertEqual("%.5f" % scores['bleu_2'], '0.11419')
        self.assertEqual("%.5f" % scores['bleu_3'], '0.08906')
        self.assertEqual("%.5f" % scores['bleu_4'], '0.07623')
        self.assertEqual("%.5f" % scores['rouge_l'], '0.12077')

if __name__ == '__main__':
    unittest.main()
