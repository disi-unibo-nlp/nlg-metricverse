import os
import unittest


class TestMetrics(unittest.TestCase):
    predictions = os.getcwd() + "/predictions"
    references = os.getcwd() + "/references"
    sources = os.getcwd() + "/sources"

    def test_abstractness(self):
        from nlgmetricverse.scoring.abstractness import abstractness
        scores = abstractness(predictions=self.predictions, references=self.references, method="read_lines")
        result = 0.29411764705882354
        self.assertEqual(scores, result, "Should be 0.29411764705882354")

    def test_average_unique_ngram(self):
        from nlgmetricverse.scoring.average_unique_ngram import average_unique_ngram
        scores = average_unique_ngram(predictions=self.predictions, references=self.references, method="read_lines")
        result = 16 / 17
        self.assertEqual(scores, result, "Should be 16/17")

    def test_readability(self):
        from nlgmetricverse.scoring.readability import readability
        scores = readability(predictions=self.predictions, references=self.references, method="read_lines")
        result = 89.9254807692308
        self.assertEqual(scores, result, "Should be 89.9254807692308")

    def test_repetitiveness(self):
        from nlgmetricverse.scoring.repetitiveness import repetitiveness
        scores = repetitiveness(predictions=self.predictions, references=self.references, method="read_lines")
        result = 6.0
        self.assertEqual(scores, result, "Should be 6.0")


if __name__ == '__main__':
    unittest.main()
