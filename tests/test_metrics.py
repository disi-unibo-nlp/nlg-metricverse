import os
import unittest

from nlgmetricverse import Nlgmetricverse


class TestMetrics(unittest.TestCase):
    predictions = os.getcwd() + "/predictions"
    references = os.getcwd() + "/references"
    sources = os.getcwd() + "/sources"

    def test_bartscore(self):
        scorer = Nlgmetricverse(metrics=["bartscore"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'bartscore': {
                'score': -1.7989066243171692,
                'model_checkpoint': 'bartscore-large-cnn',
                'model_weights': None,
                'segment_scores': False
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_bertscore(self):
        scorer = Nlgmetricverse(metrics=["bertscore"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'bertscore': {
                'score': 0.9695505797863007,
                'precision': 0.968218058347702,
                'recall': 0.970888078212738,
                'f1': 0.9695505797863007,
                'hashcode': 'roberta-large_L17_no-idf_version=0.3.11(hug_trans=4.10.3)'
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_bleu(self):
        scorer = Nlgmetricverse(metrics=["bleu"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'bleu': {
                'score': 0.3378703280802838,
                'precisions': [0.84, 0.5714285714285714, 0.35294117647058826, 0.07692307692307693],
                'brevity_penalty': 1.0,
                'length_ratio': 1.1818181818181819,
                'translation_length': 13,
                'reference_length': 11
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_bleurt(self):
        scorer = Nlgmetricverse(metrics=["bleurt"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        '''
        import nlgmetricverse
        bleurt = nlgmetricverse.load_metric("bleurt", config_name="bleurt-tiny-128")
        predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
        references = [
            ["the cat is playing on the mat.", "The cat plays on the mat."],
            ["Today is a wonderful day", "The weather outside is wonderful."]
        ]
        scores = bleurt.compute(predictions=predictions, references=references)
        '''
        result = {
            'bleurt': {
                'score': 0.25963682122528553,
                'scores': [0.47344309091567993, 0.04583055153489113],
                'checkpoint': 'bleurt-tiny-128'
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_chrf(self):
        scorer = Nlgmetricverse(metrics=["chrf"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'chrf': {
                'score': 0.44298405744188873,
                'char_order': 6,
                'word_order': 0,
                'beta': 2
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_comet(self):
        scorer = Nlgmetricverse(metrics=["comet"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        '''
        import nlgmetricverse
        comet_metric = nlgmetricverse.load_metric('comet', config_name="wmt21-cometinho-da")
        source = ["Die Katze spielt auf der Matte.", "Heute ist ein wunderbarer Tag."]
        predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
        references = [
            ["the cat is playing on the mat.", "The cat plays on the mat."],
            ["Today is a wonderful day", "The weather outside is wonderful."]
        ]
        scores = comet_metric.compute(sources=source, predictions=predictions, references=references)
        '''
        result = {
            'comet': {
                'scores': [0.6338749527931213, 0.4925243854522705],
                'samples': 0.5631996691226959
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_meteor(self):
        scorer = Nlgmetricverse(metrics=["meteor"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'meteor': {
                'score': 0.9012408828265606
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_moverscore(self):
        scorer = Nlgmetricverse(metrics=["moverscore"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'moverscore': {
                'score': 0.6407421179890045
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_nist(self):
        scorer = Nlgmetricverse(metrics=["nist"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'nist': {
                'score': 1.2580194300219625
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_rouge(self):
        scorer = Nlgmetricverse(metrics=["rouge"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            "rouge": {
                "rouge1": 0.8541458541458541,
                "rouge2": 0.5845959595959596,
                "rougeL": 0.772977022977023,
                "rougeLsum": 0.772977022977023
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_sacrebleu(self):
        scorer = Nlgmetricverse(metrics=["sacrebleu"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            "sacrebleu": {
                "score": 0.6475445426291289,
                "counts": [
                    11,
                    6,
                    3,
                    0
                ],
                "totals": [
                    13,
                    11,
                    9,
                    7
                ],
                "precisions": [
                    0.8461538461538461,
                    0.5454545454545454,
                    0.33333333333333337,
                    0.07142857142857144
                ],
                "bp": 1.0,
                "sys_len": 13,
                "ref_len": 12,
                "adjusted_precisions": [
                    1.6923076923076923,
                    1.0909090909090908,
                    0.6666666666666667,
                    0.14285714285714288
                ]
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_ter(self):
        scorer = Nlgmetricverse(metrics=["ter"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            'ter': {
                'score': 0.6307692307692307,
                'avg_num_edits': 2.5,
                'avg_ref_length': 5.75
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_wer(self):
        scorer = Nlgmetricverse(metrics=["wer"])
        scores = scorer(predictions=self.predictions, references=self.references, method="read_lines")
        result = {
            "wer": {
                "score": 1.2,
                "overall": {
                    "substitutions": 3.0,
                    "deletions": 0.125,
                    "insertions": 0.625,
                    "hits": 2.625
                }
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")


if __name__ == '__main__':
    unittest.main()
