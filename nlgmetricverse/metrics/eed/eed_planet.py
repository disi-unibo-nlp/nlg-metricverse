""" EED metric. The part of this file is adapted from rwth-i6 implementation. See
https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py for the original version """


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
    "libed": "https://github.com/rwth-i6/ExtendedEditDistance/raw/master/libEED.so",
    "ed":"https://github.com/rwth-i6/CharacTER/blob/master/ed.cpp"
}
@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EEDPlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            **kwargs,
    ):
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        
        libED_path = dl_manager.download(LIBRARY_URLS["libed"])
        self.ed_wrapper = ctypes.CDLL(libED_path)
        self.ed_wrapper.wrapper.restype = ctypes.c_float
    
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/rwth-i6/ExtendedEditDistance",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/rwth-i6/ExtendedEditDistance"],
            reference_urls=[
                "https://github.com/rwth-i6/ExtendedEditDistance",
                "https://aclanthology.org/W19-5359/",
            ],
            license=_LICENSE,
        )


    def bytes_to_int(self,bytes):
        result = 0
        for b in bytes:
            result = result * 256 + int(b)
        return result

    def eed(self,hyp, ref):
        hyp.insert(0, " ")
        hyp.append(" ")
        ref.insert(0, " ")
        ref.append(" ")
        hyp_c = (ctypes.c_ulonglong * len(hyp))()
        hyp_c[:] = [self.bytes_to_int(x.encode('utf-8')) for x in hyp]
        ref_c = (ctypes.c_ulonglong * len(ref))()
        ref_c[:] = [self.bytes_to_int(x.encode('utf-8')) for x in ref]
        alpha = 2.0
        deletion = 0.2
        insertion = 1.0
        substitution = 1.0
        rho = 0.3
        norm = len(ref_c)
        result = self.ed_wrapper.wrapper(hyp_c, ref_c, len(hyp_c), len(ref_c), 
                            ctypes.c_float(alpha), ctypes.c_float(deletion), 
                            ctypes.c_float(insertion), ctypes.c_float(substitution), ctypes.c_float(rho), 
                            norm) 
        return min(1.0, result)    

    def score(self,hyp, ref):
        scores = []
        for (h,r) in zip(hyp,ref):
            h, r = list(h), list(r)
            score = self.eed(h,r)
            scores.append(score)    
        return scores

    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        scores = self.score(predictions, references)
        return {
            "score":  float(np.mean(scores)),
        }

    def _compute_single_pred_multi_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        reduced_scores = []
        for refs, pred in zip(references, predictions):
            extended_preds = [pred for _ in range(len(refs))]
            reduced_scores.append(reduce_fn(self.score(extended_preds, refs)))
        return {
            "score": float(np.mean(reduced_scores)), "reduced_scores" : reduced_scores
        }


    def _compute_multi_pred_multi_ref(
        self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
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
              "score": float(np.mean(reduced_scores)), "reduced_scores" : reduced_scores
        }