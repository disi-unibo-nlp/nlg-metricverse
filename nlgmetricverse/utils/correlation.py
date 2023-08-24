"""
Benchmarks: WMT17 and WMT16, are the product of the Workshop on Statistical Machine 
Translation (WMT) and the Conference on Machine Translation (WMT) of 2016 and 2017.
As from the Wmt Metrics Task site, https://wmt-metrics-task.github.io/:

"This shared task will examine automatic evaluation metrics for machine translation. 
We will provide you with MT system outputs along with source text and the human reference 
translations. We are looking for automatic metric scores for translations at the system-level, 
and segment-level. We will calculate the system-level, and segment-level correlations of 
your scores with human judgements."

Link for the WMT17 Metrics Task is: https://www.statmt.org/wmt17/ and https://www.statmt.org/wmt17/metrics-task.html
Link for the WMT18 Metrics Task is: https://www.statmt.org/wmt18/ and https://www.statmt.org/wmt18/metrics-task/
"""
import bert_score
import numpy as np
import pandas as pd
import pickle as pkl

import os
import torch
import requests
import tarfile

from enum import Enum
from tqdm.auto import tqdm
from collections import defaultdict
from scipy.stats import pearsonr
from scipy.stats import pearsonr, spearmanr, kendalltau

from nlgmetricverse.utils.sys import log
from nlgmetricverse import NLGMetricverse, load_metric
from nlgmetricverse.metrics._core.utils import get_metric_bounds

wmt17_sys_to_lang_pairs = ['cs-en', 'de-en', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en']
wmt17_sys_from_lang_pairs = ['en-cs', 'en-de', 'en-lv', 'en-ru', 'en-tr', 'en-zh']
wmt17_sys_lang_pairs = wmt17_sys_to_lang_pairs + wmt17_sys_from_lang_pairs
wmt18_sys_to_lang_pairs = ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en']
wmt18_sys_from_lang_pairs = ['en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']
wmt18_sys_lang_pairs = wmt18_sys_to_lang_pairs + wmt18_sys_from_lang_pairs

def map_range(value, left_min, left_max, right_min, right_max):
    """
    This method, scales a given input value from one range to another. 
    The method calculates the spans of both the source and target ranges and 
    then scales the input value accordingly. The scaled value within the target 
    range is returned as the output. This function effectively transforms a value 
    from one range to another while maintaining its proportional position.

    :param value: The value to scale
    :param left_min: The minimum value of the source range
    :param left_max: The maximum value of the source range
    :param right_min: The minimum value of the target range
    :param right_max: The maximum value of the target range
    """
    left_span = left_max - left_min
    right_span = right_max - right_min
    value_scaled = float(value - left_min) / float(left_span)
    return right_min + (value_scaled * right_span)


def map_score_with_metric_bounds(metric, score):
    """
    This method, maps a list of scores for a specific metric to a normalized scale 
    between 0 and 1, considering metric-specific upper and lower bounds. 
    The method iterates through each score and retrieves the upper and lower bounds 
    for the given metric using the `get_metric_bounds()` function. If the bounds are 
    not the default values (1 for upper and 0 for lower), the method scales the score 
    using the `map_range()` function. If the bounds are the default values, the score 
    remains unchanged. The normalized and mapped scores are then returned as a list. 
    This process allows scores to be transformed into a consistent range for comparison 
    while respecting metric-specific constraints.

    :param metric: The metric to map the scores for
    :param score: The scores to map
    """
    mapped_score = []
    for i, single_score in enumerate(score):
        upper_bound, lower_bound = get_metric_bounds(metric)
        if upper_bound != 1 or lower_bound != 0:
            mapped_score.append(map_range(single_score, lower_bound, upper_bound, 0, 1))
        else:
            mapped_score.append(single_score)
    return mapped_score

def calculate_scores(predictions, references, metrics, human_flag):
    """
    This for loop iterates through each metric in the list of metrics and calculates 
    the metric score for each prediction-reference pair. The metric score is then
    stored in the list of individual metric scores, single_metric_scores. The list 
    of individual metric scores is then iterated through and the individual scores 
    are stored in the list of aggregated metric scores, res. If the metric is "rouge", 
    the mean of rouge1, rouge2, and rougeL scores is calculated and stored in the 
    aggregated scores list. The aggregated metric scores are then mapped to the 
    metric bounds and stored in the scores dictionary.
    """
    scores={}
    for metric in metrics:
        if human_flag:
            if not isinstance(predictions, list) and not isinstance(references, list):
                raise Exception("predictions and references must be of type list")
        single_metric_scores = []
        res = []
        single_metric_scorer = NLGMetricverse(metrics=load_metric(metric))
        for i, pred in enumerate(predictions):
            single_metric_score = single_metric_scorer(predictions=[pred], references=[references[i]])
            single_metric_scores.append(single_metric_score)
            for single_score in single_metric_score:
                if isinstance(single_metric_score[single_score], dict):
                    if metric == "rouge":
                        mean = single_metric_score[single_score]["rouge1"] + \
                            single_metric_score[single_score]["rouge2"] + single_metric_score[single_score]["rougeL"]
                        mean = mean / 3
                        res.append(mean)
                    else:
                        res.append(single_metric_score[single_score]["score"])
        scores[metric] = map_score_with_metric_bounds(metric, res)
    return scores

class CorrelationMeasures(Enum):
    """
    The Pearson correlation coefficient $\rho$ between two vectors $y$ and $\widehat{y}$ of dimension $N$ is:
    $$
    pearsonr(y, \widehat{y}) =
    \frac{
      \sum_{i}^{N} (y_{i} - \mu_{y}) \cdot (\widehat{y}_{i} - \mu_{\widehat{y}})
    }{
      \sum_{i}^{N} (y_{i} - \mu_{y})^{2} \cdot (\widehat{y}_{i} - \mu_{\widehat{y}})^{2}
    }
    $$
    where $\mu_{y}$ is the mean of $y$ and $\mu_{\widehat{y}}$ is the mean of $\widehat{y}$.
    For comparing gold values $y$ and predicted values $\widehat{y}$, Pearson correlation is equivalent to a linear regression using
    $\widehat{y}$ and a bias term to predict $y$ (https://lindeloev.github.io/tests-as-linear/).
    It is also closely related to R-squared.

    Pearson Correlation [15] measures the linear correlation between two sets of data. Spearman
    Correlation [73] assesses the monotonic relationships between two variables. Kendall’s Tau [27]
    measures the ordinal association between two measured quantities. BartScore.

    BOUNDS
    [-1, 1], where -1 is a complete negative linear correlation, +1 is a complete positive linear correlation, and 0 is no linear
    correlation at all.

    WEAKNESSES
    Pearson correlations are highly sensitive to the magnitude of the differences between the gold and predicted values. As a result,
    they are also very sensitive to outliers.

    The Spearman rank correlation coefficient between two vectors $y$ and $\widehat{y}$ of dimension $N$ is the Pearson
    coefficient with all the data mapped to their ranks.

    BOUNDS
    [-1, 1], where -1 is a complete negative linear correlation, +1 is a complete positive linear correlation, and 0 is no linear
    correlation at all.

    WEAKNESSES
    Unlike Pearson, Spearman is not sensitive to the magnitude of the differences. In fact, it's invariant under all monotonic
    rescaling, since the values are converted to ranks. This also makes it less sensitive to outliers than Pearson.
    Of course, these strengths become weaknesses in domains where the raw differences do matter. That said, in most NLU contexts,
    Spearman will be a good conservative choice for system assessment.

    The Kendall rank correlation coefficient between two vectors $y$ and $\widehat{y}$ of dimension $N$ is the number of
    concordant pairs minus the number of discordant pairs, divided by the total number of pairs. A pair is concordant if the
    ranks of $y$ and $\widehat{y}$ agree, and discordant if they do not. The Kendall coefficient is equivalent to the probability
    that the two vectors are in the same order.

    BOUNDS
    [-1, 1], where -1 is a complete negative linear correlation, +1 is a complete positive linear correlation, and 0 is no linear
    correlation at all.

    WEAKNESSES
    Kendall is less sensitive to outliers than Pearson, but more sensitive than Spearman. It is also less sensitive to monotonic
    transformations than Spearman. It is a good choice when you want to measure the ordinal association between two measured
    quantities.

    The daRR correlation coefficient between two vectors $y$ and $\widehat{y}$ of dimension $N$ is the number of concordant
    pairs minus the number of discordant pairs, divided by the total number of pairs. A pair is concordant if the ranks of $y$ and
    $\widehat{y}$ agree, and discordant if they do not. The daRR coefficient is equivalent to the probability that the two vectors
    are in the same order.

    BOUNDS
    [-1, 1], where -1 is a complete negative linear correlation, +1 is a complete positive linear correlation, and 0 is no linear
    correlation at all.

    WEAKNESSES
    daRR being a variant of Kendall, like Kendall it is less sensitive to outliers than Pearson, but more sensitive than Spearman.
    However it is more sensitive to monotonic transformations than Spearman. It is a good choice when you want to measure the ordinal
    association between two measured quantities.
    """
    Pearson = 1
    Spearman = 2
    KendallTau = 3
    daRR = 4


class Benchmarks(Enum):
    WMT17 = 1,
    WMT18 = 2


def compute_correlation(x, y, correlation_measure):
    """
    This method, calculates a correlation statistic and p-value between two input 
    arrays, `x` and `y`, using a specified correlation measure. The correlation measure 
    is chosen from a set of options enumerated by the `CorrelationMeasures` enum. 
    Depending on the selected measure, the method employs different correlation functions 
    (e.g., Pearson, Spearman, Kendall's Tau) to compute the correlation statistic and 
    associated p-value. The method then returns the calculated statistic and p-value, 
    allowing for the assessment of the correlation strength and significance between the 
    two arrays according to the chosen correlation measure.

    :param x: The first array to calculate correlation for
    :param y: The second array to calculate correlation for
    :param correlation_measure: The correlation measure to use
    """
    if correlation_measure == CorrelationMeasures.Pearson:
        statistic, pvalue = pearsonr(x, y)
    elif correlation_measure == CorrelationMeasures.Spearman:
        statistic, pvalue = spearmanr(x, y)
    elif correlation_measure == CorrelationMeasures.KendallTau:
        statistic, pvalue = kendalltau(x, y)
    elif correlation_measure == CorrelationMeasures.daRR:
        statistic, pvalue = kendalltau(x, y, variant="c")
    else:
        statistic, pvalue = pearsonr(x, y)
    return statistic, pvalue

def get_wmt17_sys_data(lang_pair):
    """
    This method, retrieves data necessary for evaluating translation system outputs 
    on the WMT17 metrics task for a given language pair. It takes the language pair 
    as input and proceeds to extract human scores from a CSV file, reference 
    translations from a text file, and gold scores from the extracted data. 
    The method gathers a list of systems participating in the task and creates 
    corresponding candidate translations. It then organizes and returns the references, 
    candidates, gold scores, and system names for further evaluation.

    :param lang_pair: The language pair to retrieve data for
    """
    first, second = lang_pair.split("-")

    human_scores = pd.read_csv(
        "wmt17/manual-evaluation/DA-syslevel.csv", delimiter=" ")

    with open("wmt17/input/wmt17-metrics-task/"
              "wmt17-submitted-data/txt/references/newstest2017-{}{}-ref.{}".format(first, second, second),
              encoding = "utf-8") as f:
        refs = f.read().strip().split("\n")

    gold_dict = dict(zip(human_scores[human_scores['LP'] == lang_pair]['SYSTEM'],
                         human_scores[human_scores['LP'] == lang_pair]['HUMAN']))
    gold_scores = []

    lang_dir = "wmt17/input/" \
               "wmt17-metrics-task/wmt17-submitted-data/" \
               "txt/system-outputs/newstest2017/{}".format(lang_pair)
    systems = [system[13:-6] for system in os.listdir(lang_dir)]

    refs *= len(systems)
    cands = []

    for system in systems:
        with open(os.path.join(lang_dir, "newstest2017.{}.{}".format(system, lang_pair)), encoding="utf-8") as f:
            cand_sys = f.read().strip().split("\n")
        gold_scores.append(gold_dict[system])

        cands += cand_sys
    return refs, cands, gold_scores, systems


def get_wmt17_sys_bert_score(lang_pair, scorer, cache=False, from_en=True, batch_size=64):
    """
    This method, calculates BERT-based scores for translation system outputs on the WMT17 metrics
    task for a given language pair and scorer. The method constructs cache filenames based on the 
    scorer's model type and the provided language pair. It checks if the scores are cached and if so, 
    it loads and returns the cached scores. If not, it retrieves reference, candidate, and gold scores 
    data, computes IDF if necessary, and then calculates the scores. Finally, it stores the calculated 
    scores and gold scores in cache files and returns the results.

    :param lang_pair: The language pair to calculate scores for
    :param scorer: The scorer to use for calculating scores
    :param cache: Whether to cache the scores or not
    :param from_en: Whether to calculate scores from English or to English
    :param batch_size: The batch size to use for calculating scores
    """
    filename = ''
    if from_en:
        if scorer.idf:
            filename = "cache_score/from_en/17/{}/wmt17_seg_from_{}_{}_idf.pkl".format(scorer.model_type,
                                                                                       *lang_pair.split('-'))
        else:
            filename = "cache_score/from_en/17/{}/wmt17_seg_from_{}_{}.pkl".format(scorer.model_type,
                                                                                   *lang_pair.split('-'))
    else:
        if scorer.idf:
            filename = "cache_score/to_en/17/{}/wmt17_seg_to_{}_{}_idf.pkl".format(scorer.model_type,
                                                                                   *lang_pair.split('-'))
        else:
            filename = "cache_score/to_en/17/{}/wmt17_seg_to_{}_{}.pkl".format(scorer.model_type, *lang_pair.split('-'))

    if os.path.exists(filename):
        with open(filename, "rb", encoding="utf-8") as f:
            return pkl.load(f)
    else:
        refs, cands, gold_scores, systems = get_wmt17_sys_data(lang_pair)
        if scorer.idf:
            scorer.compute_idf(refs)
        raw_scores = scorer.score(cands, refs, batch_size=batch_size)
        scores = [s.view(len(systems), -1).mean(dim=-1) for s in raw_scores]

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb", encoding="utf-8") as f:
            pkl.dump((scores, gold_scores), f)

    return scores, gold_scores


def get_wmt17_sys_results(
        model=None,
        log_file="wmt17_log.csv",
        idf=False,
        batch_size=64,
        lang_pairs=None
):
    """
    This method, calculates BERT-based scores for translation system outputs
    on the WMT17 metrics task. If no model or language pairs are provided, 
    default values are used. The method initializes BERTScorer and computes 
    scores for precision, recall, and F1 for each language pair. It then 
    calculates average scores and logs the results into a CSV file. The method 
    iterates through different model types and language pairs, generating and 
    recording BERT-based evaluation scores.

    :param model: The model to use for calculating scores
    :param log_file: The file to log the results from
    :param idf: Whether to use IDF or not
    :param batch_size: The batch size to use for calculating scores
    :param lang_pairs: The language pairs to calculate scores for
    """
    if model is None:
        model = ["microsoft/deberta-xlarge-mnli"]
    if lang_pairs is None:
        lang_pairs = wmt17_sys_from_lang_pairs
    torch.set_grad_enabled(False)

    header = 'model_type'
    for lang_pair in lang_pairs + ['avg']:
        header += f',{lang_pair}'
    log(header)
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding="utf-8") as f:
            log(header, file=f)

    log(model)
    for model_type in model:
        scorer = bert_score.scorer.BERTScorer(model_type=model_type, idf=idf)
        results = defaultdict(dict)
        for lang_pair in tqdm(lang_pairs):
            scores, gold_scores = get_wmt17_sys_bert_score(lang_pair, scorer, batch_size=batch_size, cache=True,
                                                           from_en=False)
            for s, name in zip(scores, ["P", "R", "F"]):
                results[lang_pair][f"{model_type} {name}"] = np.mean(pearsonr(gold_scores, s)[0])

        for name in ["P", "R", "F"]:
            temp = []
            for lang_pair in lang_pairs:
                temp.append(results[lang_pair][f"{model_type} {name}"])
            results["avg"][f"{model_type} {name}"] = np.mean(temp)

            msg = f"{model_type} {name} (idf)" if idf else f"{model_type} {name}"
            for lang_pair in lang_pairs + ['avg']:
                msg += f",{results[lang_pair][f'{model_type} {name}']}"
            log(msg)
            with open(log_file, "a", encoding="utf-8") as f:
                log(msg, file=f)

        del scorer


def wmt17_download_data():
    """
    This method, downloads and extracts data for the WMT17 metrics task. 
    It begins by setting up the necessary directory structure and navigating 
    to the appropriate paths. Then, it checks if the required files and 
    directories exist, and if not, it proceeds to download the necessary 
    data from a specified URL. The method handles the downloading and extraction 
    of both the main archive and a sub-archive.
    """
    starting_dir = os.getcwd()
    directory = "wmt17"
    parent_dir = os.path.curdir
    path = os.path.join(parent_dir, directory)
    if not os.path.isdir('./wmt17'):
        directory = "wmt17"
        parent_dir = os.path.curdir
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    os.chdir(path)
    if not os.path.isfile('./wmt17.tgz'):
        url = 'http://ufallab.ms.mff.cuni.cz/~bojar/wmt17-metrics-task-package.tgz'
        r = requests.get(url, allow_redirects=True)
        open('wmt17.tgz', 'wb', encoding="utf-8").write(r.content)
    if not os.path.isdir('input'):
        tar = tarfile.open("wmt17.tgz", encoding="utf-8")
        tar.extractall()
        tar.close()
    directory = "input"
    parent_dir = os.path.curdir
    path = os.path.join(parent_dir, directory)
    os.chdir(path)
    if not os.path.isdir('wmt17-metrics-task'):
        tar = tarfile.open("wmt17-metrics-task.tgz", encoding="utf-8")
        tar.extractall()
        tar.close()
    os.chdir(starting_dir)


def get_wmt18_sys_data(lang_pair):
    """
    This method, retrieves data necessary for evaluating translation system outputs 
    on the WMT18 metrics task for a given language pair. It takes the language pair 
    as input and proceeds to extract human scores from a CSV file, reference 
    translations from a text file, and gold scores from the extracted data. 
    The method gathers a list of systems participating in the task and creates 
    corresponding candidate translations. It then organizes and returns the references, 
    candidates, gold scores, and system names for further evaluation.

    :param lang_pair: The language pair to retrieve data for
    """
    first, second = lang_pair.split("-")

    human_scores = pd.read_csv(
        "wmt18/manual-evaluation/DA-syslevel.csv", delimiter=" ")

    with open("wmt18/input/wmt18-metrics-task-nohybrids/"
              "references/newstest2018-{}{}-ref.{}".format(first, second, second),
              encoding = "utf-8") as f:
        refs = f.read().strip().split("\n")

    gold_dict = dict(zip(human_scores[human_scores['LP'] == lang_pair]['SYSTEM'],
                         human_scores[human_scores['LP'] == lang_pair]['HUMAN']))
    gold_scores = []

    lang_dir = "wmt18/input/" \
               "wmt18-metrics-task-nohybrids/wmt17-submitted-data/" \
               "system-outputs/newstest2017/{}".format(lang_pair)
    systems = [system[13:-6] for system in os.listdir(lang_dir)]

    refs *= len(systems)
    cands = []

    for system in systems:
        with open(os.path.join(lang_dir, "newstest2018.{}.{}".format(system, lang_pair)), encoding="utf-8") as f:
            cand_sys = f.read().strip().split("\n")
        gold_scores.append(gold_dict[system])

        cands += cand_sys
    return refs, cands, gold_scores, systems


def get_wmt18_sys_bert_score(lang_pair, scorer, cache=False, from_en=True, batch_size=64):
    """
    This method, calculates BERT-based scores for translation system outputs on the WMT18 metrics
    task for a given language pair and scorer. The method constructs cache filenames based on the 
    scorer's model type and the provided language pair. It checks if the scores are cached and if so, 
    it loads and returns the cached scores. If not, it retrieves reference, candidate, and gold scores 
    data, computes IDF if necessary, and then calculates the scores. Finally, it stores the calculated 
    scores and gold scores in cache files and returns the results.

    :param lang_pair: The language pair to calculate scores for
    :param scorer: The scorer to use for calculating scores
    :param cache: Whether to cache the scores or not
    :param from_en: Whether to calculate scores from English or to English
    :param batch_size: The batch size to use for calculating scores
    """
    filename = ''
    if from_en:
        if scorer.idf:
            filename = "cache_score/from_en/18/{}/wmt18_seg_from_{}_{}_idf.pkl".format(scorer.model_type,
                                                                                       *lang_pair.split('-'))
        else:
            filename = "cache_score/from_en/18/{}/wmt18_seg_from_{}_{}.pkl".format(scorer.model_type,
                                                                                   *lang_pair.split('-'))
    else:
        if scorer.idf:
            filename = "cache_score/to_en/18/{}/wmt18_seg_to_{}_{}_idf.pkl".format(scorer.model_type,
                                                                                   *lang_pair.split('-'))
        else:
            filename = "cache_score/to_en/18/{}/wmt18_seg_to_{}_{}.pkl".format(scorer.model_type, *lang_pair.split('-'))

    if os.path.exists(filename):
        with open(filename, "rb", encoding="utf-8") as f:
            return pkl.load(f)
    else:
        refs, cands, gold_scores, systems = get_wmt18_sys_data(lang_pair)
        if scorer.idf:
            scorer.compute_idf(refs)
        raw_scores = scorer.score(cands, refs, batch_size=batch_size)
        scores = [s.view(len(systems), -1).mean(dim=-1) for s in raw_scores]

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb", encoding="utf-8") as f:
            pkl.dump((scores, gold_scores), f)

    return scores, gold_scores


def get_wmt18_sys_results(
        model=None,
        log_file="wmt18_log.csv",
        idf=False,
        batch_size=64,
        lang_pairs=None
):
    """
    This method, calculates BERT-based scores for translation system outputs
    on the WMT18 metrics task. If no model or language pairs are provided, 
    default values are used. The method initializes BERTScorer and computes 
    scores for precision, recall, and F1 for each language pair. It then 
    calculates average scores and logs the results into a CSV file. The method 
    iterates through different model types and language pairs, generating and 
    recording BERT-based evaluation scores.

    :param model: The model to use for calculating scores
    :param log_file: The file to log the results from
    :param idf: Whether to use IDF or not
    :param batch_size: The batch size to use for calculating scores
    :param lang_pairs: The language pairs to calculate scores for
    """
    if model is None:
        model = ["microsoft/deberta-xlarge-mnli"]
    if lang_pairs is None:
        lang_pairs = wmt18_sys_from_lang_pairs
    torch.set_grad_enabled(False)

    header = 'model_type'
    for lang_pair in lang_pairs + ['avg']:
        header += f',{lang_pair}'
    log(header)
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding="utf-8") as f:
            log(header, file=f)

    log(model)
    for model_type in model:
        scorer = bert_score.scorer.BERTScorer(model_type=model_type, idf=idf)
        results = defaultdict(dict)
        for lang_pair in tqdm(lang_pairs):
            scores, gold_scores = get_wmt18_sys_bert_score(lang_pair, scorer, batch_size=batch_size, cache=True,
                                                           from_en=False)
            for s, name in zip(scores, ["P", "R", "F"]):
                results[lang_pair][f"{model_type} {name}"] = np.mean(pearsonr(gold_scores, s)[0])

        for name in ["P", "R", "F"]:
            temp = []
            for lang_pair in lang_pairs:
                temp.append(results[lang_pair][f"{model_type} {name}"])
            results["avg"][f"{model_type} {name}"] = np.mean(temp)

            msg = f"{model_type} {name} (idf)" if idf else f"{model_type} {name}"
            for lang_pair in lang_pairs + ['avg']:
                msg += f",{results[lang_pair][f'{model_type} {name}']}"
            log(msg)
            with open(log_file, "a", encoding="utf-8") as f:
                log(msg, file=f)

        del scorer


def wmt18_download_data():
    """
    This method, downloads and extracts data for the WMT18 metrics task. 
    It begins by setting up the necessary directory structure and navigating 
    to the appropriate paths. Then, it checks if the required files and 
    directories exist, and if not, it proceeds to download the necessary 
    data from a specified URL. The method handles the downloading and extraction 
    of both the main archive and a sub-archive. Finally, it returns to the initial 
    directory after completing the data retrieval process.
    """
    starting_dir = os.getcwd()
    directory = "wmt18"
    parent_dir = os.path.curdir
    path = os.path.join(parent_dir, directory)
    if not os.path.isdir('./wmt18'):
        directory = "wmt18"
        parent_dir = os.path.curdir
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    os.chdir(path)
    if not os.path.isfile('./wmt18.tgz'):
        url = 'http://ufallab.ms.mff.cuni.cz/~bojar/wmt18-metrics-task-package.tgz'
        r = requests.get(url, allow_redirects=True)
        open('wmt18.tgz', 'wb', encoding="utf-8").write(r.content)
    if not os.path.isdir('input'):
        tar = tarfile.open("wmt18.tgz", encoding="utf-8")
        tar.extractall()
        tar.close()
    directory = "input"
    parent_dir = os.path.curdir
    path = os.path.join(parent_dir, directory)
    os.chdir(path)
    if not os.path.isfile('./wmt18-metrics-task-nohybrids.tgz'):
        url = 'http://ufallab.ms.mff.cuni.cz/~bojar/wmt18/wmt18-metrics-task-nohybrids.tgz'
        r = requests.get(url, allow_redirects=True)
        open('wmt18-metrics-task-nohybrids.tgz', 'wb', encoding="utf-8").write(r.content)
    if not os.path.isdir('wmt18-metrics-task-nohybrids'):
        tar = tarfile.open("wmt18-metrics-task-nohybrids.tgz", encoding="utf-8")
        tar.extractall()
        tar.close()
    os.chdir(starting_dir)