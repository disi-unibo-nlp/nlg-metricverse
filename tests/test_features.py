import os
import unittest


class TestMetrics(unittest.TestCase):
    predictions = os.getcwd() + "/predictions"
    references = os.getcwd() + "/references"
    sources = os.getcwd() + "/sources"

    def test_metrics_correlation(self):
        from nlgmetricverse.correlations import metrics_correlation as mc
        scores = mc.pearson_and_spearman(
            predictions=self.predictions,
            references=self.references,
        )
        result = [0.25473327402089213, 0.38700274580351945, 0.5217391304347826]
        self.assertEqual(scores, result, "Should be 0.29411764705882354")


if __name__ == '__main__':
    unittest.main()