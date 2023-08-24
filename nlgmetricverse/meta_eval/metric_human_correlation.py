import numpy as np

from nlgmetricverse.utils.correlation import *
from nlgmetricverse import NLGMetricverse, load_metric
from nlgmetricverse.visualization.correlation_visualization import mhc_visual


def metric_human_correlation(
        predictions,
        references,
        metrics,
        human_scores,
        correlation_measures=[CorrelationMeasures.Pearson]
):
    """
    Calculates the correlation between human scores and those of the metrics, with the possibility of choosing
    between several correlation techniques.

    :param predictions: List of predictions
    :param references: List of references
    :param metrics: List of metrics
    :param human_scores: Can be a list of personal scores or can choose a public benchmark, WMT17 or WMT18
    :param correlation_measures: The correlation technique to apply
    """
    if not isinstance(human_scores, list):
        if human_scores == Benchmarks.WMT17:
            # Generate ad DB with human scores from WMT
            # print("Downloading wmt17 data, first time might take a bit...")
            wmt17_download_data()
            # print("Download completed")
            # print("Evaluating...")
            data = get_wmt17_sys_results()
            # print("Evaluation completed, you can see the results in the 'wmt17' folder")
            return data
        elif human_scores == Benchmarks.WMT18:
            wmt18_download_data()
            data = get_wmt18_sys_results()
            return data
    else:
        scores = calculate_scores(predictions, references, metrics, human_flag=True)

        results = []
        for metric in metrics:
            metric_scores = []
            """
            This for loop iterates through each correlation measure in the list of correlation
            measures. The correlation statistic and p-value are calculated between the metric
            scores and human scores using the compute_correlation function. The correlation
            statistic is then mapped to a range between 0 and 1 and appended to the results
            list. The mean of the metric scores is calculated and appended to the results list.
            """
            for correlation_measure in tqdm(correlation_measures, desc="Calculating correlation " + metric):
                statistic, pvalue = compute_correlation(scores[metric], human_scores, correlation_measure)
                statistic = map_range(statistic, -1, 1, 0, 1)
                metric_scores.append(statistic)
            results.append(np.mean(metric_scores))

        mhc_visual(metrics, results)

        return results