
""" WMD metric. The part of this file is adapted from Gensim's WMD implementation. See
https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html """
import evaluate
import numpy as np
from typing import Callable, Dict
from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import requirement_message
try:
    from itertools import izip as zip
except ImportError:
    pass
_LICENSE= """ """
_DESCRIPTION = """ """
_CITATION = """ """



_KWARGS_DESCRIPTION = """

"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class WMDPlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            model_name = "word2vec-google-news-300",
            enable_stop_words = True,
            **kwargs,
    ):
        self.model_name=model_name
        self.enable_stop_words = enable_stop_words
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of WMD score from the implementation
        of WMD computation from gensim library. 
        """
        try:
            import gensim.downloader as api
        except:
             raise ModuleNotFoundError(requirement_message(path="WMD", package_name="gensim"))
        else:
            self.model = api.load(self.model_name)
            if self.enable_stop_words:
                import nltk
                from nltk import download
                from nltk.corpus import stopwords
                try:
                    nltk.data.find('corpora/stopwords.zip')
                except:
                    download('stopwords')  # Download stopwords list.
                self.stop_words = stopwords.words('english') 
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/mkusner/wmd",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html"],
            reference_urls=[
                "https://proceedings.mlr.press/v37/kusnerb15.pdf",
                "https://ieeexplore.ieee.org/document/5459199",
                "https://link.springer.com/chapter/10.1007/978-3-540-88690-7_37"
            ],
            license=_LICENSE,
        )
    def preprocess(self,sentence):
        return [w for w in sentence.lower().split() if w not in self.stop_words]

    def wmd_scorer(self, predictions, references):
        distances = []
        for pred, ref in zip(predictions, references):
            if self.enable_stop_words:
                pred = self.preprocess(pred)
                ref = self.preprocess(ref)
            distances.append(self.model.wmdistance(pred, ref))
        return distances

    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        distances = self.wmd_scorer(predictions, references)
        return {
            "avg_distance":  float(np.mean(distances)), "distances": distances
        }

    def _compute_single_pred_multi_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        reduced_distances = []
        for refs, pred in zip(references, predictions):
            extended_preds = [pred for _ in range(len(refs))]
            reduced_distances.append(reduce_fn(self.wmd_scorer(extended_preds, refs)))
        return {
            "avg_distance":  float(np.mean(reduced_distances)), "reduced_distances" : reduced_distances
        }


    def _compute_multi_pred_multi_ref(
        self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):

        reduced_distances = []
        for preds, refs in zip(predictions, references):
            scores= []
            for pred in preds:
                scores.append(self._compute_single_pred_multi_ref(
                        predictions=[pred],
                        references=[refs],
                        reduce_fn=reduce_fn,
                )["reduced_distances"])
            reduced_distances.append(reduce_fn(scores))
        return {
             "avg_distance": float(np.mean(reduced_distances)), "reduced_distances" : reduced_distances
        }