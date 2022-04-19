import os
import unittest


class TestMetrics(unittest.TestCase):
    predictions = os.getcwd() + "/correlation/predictions"
    references = os.getcwd() + "/correlation/references"

    def test_metrics_correlation_pearson(self):
        from nlgmetricverse.meta_eval import metrics_correlation as mc
        scores = mc.pearson_and_spearman(
            predictions=self.predictions,
            references=self.references,
            technique="pearson"
        )
        result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.5340850508745185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.9062845055092298, -0.6846425867625664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.2982064827099795, 0.5324218033708304, -0.0958378620972638, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.27046446777853295, 0.6320767380808305, -0.07372553106431701, 0.9739400052860063, 0.0, 0.0, 0.0,
                   0.0],
                  [0.40152378487814244, 0.40593070978437523, -0.0019071139891049804, 0.989516310071872,
                   0.9406715653956683, 0.0, 0.0, 0.0],
                  [0.48944893953629076, -0.9913059568081012, 0.6054951373901206, -0.5000269126704459,
                   -0.625187050954443, -0.36971465206758247, 0.0, 0.0],
                  [0.5664278999522041, -0.9882350273552676, 0.65475102902782, -0.422916796512204, -0.5496775832083525,
                   -0.28775275072651474, 0.9954027871841233, 0.0]]
        self.assertEqual(scores.tolist(), result, "Should be ...")

    def test_metrics_correlation_spearman(self):
        from nlgmetricverse.meta_eval import metrics_correlation as mc
        scores = mc.pearson_and_spearman(
            predictions=self.predictions,
            references=self.references,
            technique="spearman"
        )
        result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.7999999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.7999999999999999, -0.39999999999999997, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.39999999999999997, 0.0, 0.19999999999999998, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.19999999999999998, 0.39999999999999997, 0.39999999999999997, 0.7999999999999999, 0.0, 0.0, 0.0,
                   0.0],
                  [0.39999999999999997, 0.0, 0.19999999999999998, 1.0, 0.7999999999999999, 0.0, 0.0, 0.0],
                  [0.7999999999999999, -1.0, 0.39999999999999997, 0.0, -0.39999999999999997, 0.0, 0.0, 0.0],
                  [0.7999999999999999, -1.0, 0.39999999999999997, 0.0, -0.39999999999999997, 0.0, 1.0, 0.0]]
        self.assertEqual(scores.tolist(), result, "Should be ...")

    def test_metrics_correlation_kendalltau(self):
        from nlgmetricverse.meta_eval import metrics_correlation as mc
        scores = mc.pearson_and_spearman(
            predictions=self.predictions,
            references=self.references,
            technique="kendalltau"
        )
        result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.6666666666666669, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.6666666666666669, -0.3333333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3333333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.3333333333333334, 0.3333333333333334, 0.6666666666666669, 0.0, 0.0, 0.0, 0.0],
                  [0.3333333333333334, 0.0, 0.0, 1.0, 0.6666666666666669, 0.0, 0.0, 0.0],
                  [0.6666666666666669, -1.0, 0.3333333333333334, 0.0, -0.3333333333333334, 0.0, 0.0, 0.0],
                  [0.6666666666666669, -1.0, 0.3333333333333334, 0.0, -0.3333333333333334, 0.0, 1.0, 0.0]]
        self.assertEqual(scores.tolist(), result, "Should be ...")


if __name__ == '__main__':
    unittest.main()
