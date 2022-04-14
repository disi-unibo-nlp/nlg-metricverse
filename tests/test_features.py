import os
import unittest


class TestMetrics(unittest.TestCase):
    predictions = os.getcwd() + "/correlation/predictions"
    references = os.getcwd() + "/correlation/references"

    def test_metrics_correlation(self):
        from nlgmetricverse.correlations import metrics_correlation as mc
        scores = mc.pearson_and_spearman(
            predictions=self.predictions,
            references=self.references,
        )
        self.assertEqual(scores, "", "Should be ...")


if __name__ == '__main__':
    unittest.main()
