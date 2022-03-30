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
                'score': -1.8933048248291016,
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
                'score': 0.9473767280578613,
                'precision': 0.9467201232910156,
                'recall': 0.9480388164520264,
                'f1': 0.9473767280578613,
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
                'score': 0.42370250917168295,
                'precisions': [0.8823529411764706, 0.6428571428571429, 0.45454545454545453, 0.125],
                'brevity_penalty': 1.0, 'length_ratio': 1.0, 'translation_length': 11, 'reference_length': 11
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_bleurt(self):
        import nlgmetricverse
        bleurt = nlgmetricverse.load_metric("bleurt", config_name="bleurt-tiny-128")
        predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
        references = [
            ["the cat is playing on the mat.", "The cat plays on the mat."],
            ["Today is a wonderful day", "The weather outside is wonderful."]
        ]
        scores = bleurt.compute(predictions=predictions, references=references)
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
                'score': 0.29778203723986857,
                'char_order': 6,
                'word_order': 0,
                'beta': 2
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_comet(self):
        import nlgmetricverse
        comet_metric = nlgmetricverse.load_metric('comet', config_name="wmt21-cometinho-da")
        source = ["Die Katze spielt auf der Matte.", "Heute ist ein wunderbarer Tag."]
        predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
        references = [
            ["the cat is playing on the mat.", "The cat plays on the mat."],
            ["Today is a wonderful day", "The weather outside is wonderful."]
        ]
        scores = comet_metric.compute(sources=source, predictions=predictions, references=references)
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
                'score': 0.727184593644221
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
                "rouge1": 0.7783882783882783,
                "rouge2": 0.5925324675324675,
                "rougeL": 0.7426739926739926,
                "rougeLsum": 0.7426739926739926
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
                "score": 0.32377227131456443,
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
                "sys_len": 11,
                "ref_len": 12,
                "adjusted_precisions": [
                  0.8461538461538461,
                  0.5454545454545454,
                  0.33333333333333337,
                  0.07142857142857144
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
                'score': 0.5307692307692308,
                'avg_num_edits': 2.75,
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
                "score": 1.0,
                "overall": {
                    "substitutions": 2.8333333333333335,
                    "deletions": 0.5,
                    "insertions": 0.16666666666666666,
                    "hits": 2.6666666666666665
                }
            },
            'empty_items': 0,
            'total_items': 2
        }
        self.assertEqual(scores, result, "Should be result")

    def test_abstractness(self):
        from nlgmetricverse.abstractness import abstractness
        scores = abstractness(predictions=self.predictions, references=self.references, method="read_lines")
        result = 0.29411764705882354
        self.assertEqual(scores, result, "Should be 0.29411764705882354")

    def test_repetitiveness(self):
        from nlgmetricverse.repetitiveness import repetitiveness
        scores = repetitiveness(predictions=self.predictions, references=self.references, method="read_lines")
        result = 6.0
        self.assertEqual(scores, result, "Should be 6.0")


if __name__ == '__main__':
    unittest.main()
