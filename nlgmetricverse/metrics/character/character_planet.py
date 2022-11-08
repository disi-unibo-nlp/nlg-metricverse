import ctypes
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

LIBRARY_URLS = {
    "libed": "https://github.com/rwth-i6/CharacTER/raw/master/libED.so",
    "ed":"https://github.com/rwth-i6/CharacTER/blob/master/ed.cpp"
}
@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CharacTERPlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            **kwargs,
    ):
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of characTER score from the implementation
        of characTER computation from rwth-i6/CharacTER. The code is sourced from a specific
        commit on the master branch, in order to keep things stable. See
        https://github.com/rwth-i6/CharacTER/blob/master/CharacTER.py   

        """
        global Levenshtein
        try:
            import Levenshtein
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="CharacTER", package_name="python-Levenshtein"))
        else:
            characTER_source = (
                "https://raw.githubusercontent.com/rwth-i6/CharacTER/master/CharacTER.py"
            )
            libED_path = dl_manager.download(LIBRARY_URLS["libed"])
            print(libED_path)
            self.ed_wrapper = ctypes.CDLL(libED_path)
            self.ed_wrapper.wrapper.restype = ctypes.c_float
            self.external_module_path = dl_manager.download(characTER_source)
            self.CTERScorer = self._get_external_resource("CharacTER", attr="cer")
        
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/rwth-i6/CharacTER",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/rwth-i6/CharacTER"],
            reference_urls=[
                "https://github.com/rwth-i6/CharacTER",
                "https://github.com/rwth-i6/CharacTER/blob/master/WMT2016_CharacTer.pdf",
            ],
            license=_LICENSE,
        )
    def cer_scorer(self, predictions: EvaluationInstance,references: EvaluationInstance):
        scores = []
        for index, (hyp, ref) in enumerate(zip(references, predictions), start=1):
            ref, hyp = ref.split(), hyp.split()
            score = self.CTERScorer(hyp, ref, self.ed_wrapper)
            scores.append(score)
        return scores

    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            batch_size: int = 4,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        scores = self.cer_scorer(predictions, references)
        return {
            "score":  float(np.mean(scores)),
        }

    def _compute_single_pred_multi_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            batch_size: int = 4,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        reduced_scores = []
        for refs, pred in zip(references, predictions):
            extended_preds = [pred for _ in range(len(refs))]
            reduced_scores.append(reduce_fn(self.cer_scorer(extended_preds, refs)))
        return {
            "score": sum(reduced_scores) / len(reduced_scores), "reduced_scores" : reduced_scores
        }

    def _compute_multi_pred_multi_ref(
        self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            batch_size: int = 4,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):

        reduced_scores = []
        for preds, refs in zip(predictions, references):
            scores= []
            for pred in preds:
                scores.append(self._compute_single_pred_multi_ref(
                        predictions=[pred],
                        references=[refs],
                        reduce_fn=reduce_fn,
                )["reduced_scores"])
            reduced_scores.append(reduce_fn(scores))
        return {
             "score": sum(reduced_scores) / len(reduced_scores), "reduced_scores" : reduced_scores
        }