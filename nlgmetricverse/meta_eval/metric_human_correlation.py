import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

import nlgmetricverse
from nlgmetricverse import data_loader
from nlgmetricverse.utils.correlation import *
from nlgmetricverse.utils.get_wmt17_sys_results import get_wmt17_sys_results, download_data


def metric_human_correlation(
        predictions,
        references,
        metrics=None,
        personal_scores=None,
        method="read_lines",
        technique="pearson",
):
    if metrics is None:
        metrics = [
            "bleu",
            "chrf",
            "meteor",
            "rouge",
            "sacrebleu",
            "ter"
        ]
    if personal_scores is None:
        # Generate ad DB with human scores from WMT
        # print("Downloading wmt17 data, first time might take a bit...")
        download_data()
        # print("Download completed")
        # print("Evaluating...")
        data = get_wmt17_sys_results()
        # print("Evaluation completed, you can see the results in the 'wmt17' folder")
        return data
    else:
        # Using personal scores
        results = {}
        for metric in metrics:
            if not isinstance(predictions, list) and not isinstance(references, list):
                dataLoader = data_loader.DataLoader(predictions=predictions, references=references, method=method)
                predictions = dataLoader.get_predictions()
                references = dataLoader.get_references()

            score = scores_single_metric(metric=metric, predictions=predictions, references=references)

            mapped_score = []
            for i, single_score in enumerate(score):
                if metric == "bartscore":
                    mapped_score.append(map_range(single_score, -7, 0, 0, 1))
                elif metric == "nist":
                    mapped_score.append(map_range(single_score, 0, 10, 0, 1))
                elif metric == "perplexity":
                    mapped_score.append(map_range(single_score, float('inf'), 1, 0, 1))
                else:
                    mapped_score.append(single_score)

            if technique == "pearson":
                results[metric] = pearsonr(mapped_score, personal_scores)[0]
            elif technique == "spearman":
                results[metric] = spearmanr(mapped_score, personal_scores)[0]
            elif technique == "kendalltau":
                results[metric] = kendalltau(mapped_score, personal_scores)[0]
            results[metric] = map_range(results[metric], -1, 1, 0, 1)

        plt.bar(list(results.keys()), results.values(), color=["tab:blue", "tab:orange", "tab:green", "tab:red",
                                                               "tab:purple", "tab:brown", "tab:pink", "tab:grey",
                                                               "tab:olive", "tab:cyan"])
        plt.title("Metric-human correlation")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        return results
