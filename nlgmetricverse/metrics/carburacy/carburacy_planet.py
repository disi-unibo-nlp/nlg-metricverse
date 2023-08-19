# coding=utf-8

""" Carburacy metric. """

import evaluate
import math
import nltk
import numpy as np
from typing import Callable

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration


_CITATION = """
@inproceedings{
  title = {Carburacy: summarization models tuning and comparison in eco-sustainable 
           regimes with a novel carbon-aware accuracy},
  author = {Gianluca Moro, Luca Ragazzi, Lorenzo Valgimigli},
  month = june,
  year = {2023},
  publisher = {Proceedings of the AAAI Conference on Artificial Intelligence},
  url = {https://scholar.google.com/citations?view_op=view_citation&hl=it&user=h6jngsQAAAAJ&citation_for_view=h6jngsQAAAAJ:UeHWp8X0CEIC},
  pages = {14417-14425},
}
"""

_DESCRIPTION = """
The Carbon-aware accuracy modeling both th AS model effectiveness and eco-sustainability: 

Where C is the kg of CO2 emissions produced by the model to process a single instance x at inference time, 
alpha and beta are trade-off hyperparameters.
"""

_KWARGS_DESCRIPTION = """
Args:
    score: The R value of the prediction.
    co2_val: The CO2 value of the prediction.
Returns:
    carburacy: The average carburacy of the predictions.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("carburacy"))
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "carburacy": { "score": 0.95 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CarburacyPlanet(MetricForLanguageGeneration):
    def _info(self):
        """
        Returns metadata about the metric.

        Returns:
            MetricInfo: An object containing metadata about the metric.
        """
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=[""],
            reference_urls=[
                ""
            ],
        )

    def postprocess_text(predictions, references):
        preds = [pred.strip() for pred in predictions]
        labels = [label.strip() for label in references]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def get_rouge_scores(rouge, predictions, references):
        predictions, references = CarburacyPlanet.postprocess_text(predictions, references)
        result_rouge = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result_rouge.items()}
        return result

    def _compute_single_pred_single_ref(
            self, predictions, references, reduce_fn: Callable = None, **kwargs
    ):
        """
        Computes the carburacy score for a single predicted text and a single reference text.

        Args:
            predictions: A parameter containing a single text sample for prediction.
            references: A parameter containing a single text sample for reference.
            reduce_fn (Callable): A function to use for reducing the carburacy scores across multiple examples.
        """
        rouge = evaluate.load("rouge")
        score = CarburacyPlanet.get_rouge_scores(rouge, predictions, references)
        rouge_1 = score['rouge1']
        rouge_2 = score['rouge2']
        rouge_3 = score['rougeLsum']
        r_val = np.mean([rouge_1, rouge_2, rouge_3]) / (1 + (np.var(np.array([rouge_1/100, rouge_2/100, rouge_3/100]),dtype=np.float64)))
        co2_val = kwargs.get("co2_val")
        if co2_val is not np.nan:
            _,result,_ = self._compute_carburacy(r_val, None, co2_val)
        print(result)
        return {"score": round(result * 100, 2)}
    
    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None, **kwargs
    ):
        """
        Computes the carburacy score for a single predicted text and a multi reference text.

        Args:
            predictions: A parameter containing a single text sample for prediction.
            references: A parameter containing multiple text sample for reference.
            reduce_fn (Callable): A function to use for reducing the carburacy scores across multiple examples.
        """
        rouge = evaluate.load("rouge")
        scores = []
        for i, pred in enumerate(predictions):
            pred = str(pred)
            refs = [str(ref).strip() for ref in references[i]]
            pred, refs = CarburacyPlanet.postprocess_text([pred], refs)
            result_rouge = rouge.compute(predictions=pred, references=refs, use_stemmer=True)
            result = {k: round(v * 100, 2) for k, v in result_rouge.items()}
            scores.append(result)
        rouge_1 = np.mean([score['rouge1'] for score in scores])
        rouge_2 = np.mean([score['rouge2'] for score in scores])
        rouge_3 = np.mean([score['rougeLsum'] for score in scores])
        r_val = np.mean([rouge_1, rouge_2, rouge_3]) / (1 + (np.var(np.array([rouge_1/100, rouge_2/100, rouge_3/100]),dtype=np.float64)))
        co2_val = kwargs.get("co2_val")
        if co2_val is not np.nan:
            _,result,_ = self._compute_carburacy(r_val, None, co2_val)
        return {"score": round(result * 100, 2)}

    def _compute_multi_pred_multi_ref(
            self, predictions, references, reduce_fn: Callable = None, **kwargs
    ):
        """
        Computes the carburacy score for a multi predicted text and a multi reference text.

        Args:
            predictions: A parameter containing multiple text sample for prediction.
            references: A parameter containing multiple text sample for reference.
            reduce_fn (Callable): A function to use for reducing the carburacy scores across multiple examples.
        """
        rouge = evaluate.load("rouge")
        predictions = [pred for sublist in predictions for pred in sublist]
        references = [ref for sublist in references for ref in sublist]
        score = CarburacyPlanet.get_rouge_scores(rouge, predictions, references)
        rouge_1 = score['rouge1']
        rouge_2 = score['rouge2']
        rouge_3 = score['rougeLsum']
        r_val = np.mean([rouge_1, rouge_2, rouge_3]) / \
            (1 + (np.var(np.array([rouge_1/100, rouge_2/100, rouge_3/100]),dtype=np.float64)))
        co2_val = kwargs.get("co2_val")
        if co2_val is not np.nan:
            _,result,_ = self._compute_carburacy(r_val, None, co2_val)
        return {"score": round(result * 100, 2)}

    @staticmethod
    def _compute_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
        carburacy_train = None
        if emission_train is not None:
            carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
        carburacy_test = None
        if emission_test is not None:
            emission_test = float(emission_test)
            carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
        carburacy = None
        if carburacy_train is not None and carburacy_test is not None:
            carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
        return carburacy_train, carburacy_test, carburacy