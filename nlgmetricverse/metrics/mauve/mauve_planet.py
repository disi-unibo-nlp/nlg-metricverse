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
class MauvePlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            p_features=None,
            q_features=None,
            p_tokens=None,
            q_tokens=None,
            num_buckets: str='auto',
            pca_max_data: int=-1,
            kmeans_explained_var: float=0.9,
            kmeans_num_redo: int=5,
            kmeans_max_iter: int=500,
            featurize_model_name: str="gpt2-large",
            device_id: int=-1,
            max_text_length: int=1024,
            divergence_curve_discretization_size: int=25,
            mauve_scaling_factor: int=5,
            _seed = 25,
            verbose: bool=True,
            **kwargs,
    ):

        self.p_features=p_features
        self.q_features=q_features
        self.p_tokens=p_tokens
        self.q_tokens=q_tokens
        self.num_buckets =num_buckets
        self.pca_max_data=pca_max_data
        self.kmeans_explained_var=kmeans_explained_var
        self.kmeans_num_redo=kmeans_num_redo
        self.kmeans_max_iter=kmeans_max_iter
        self.featurize_model_name=featurize_model_name
        self.device_id=device_id
        self.max_text_length=max_text_length
        self.divergence_curve_discretization_size=divergence_curve_discretization_size
        self.mauve_scaling_factor=mauve_scaling_factor
        self.verbose=verbose
        self._seed=_seed
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Import the computation of mauve score from mauve-text library.
        """
        global mauve
        try:
            import mauve
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="Mauve", package_name="mauve-text"))
          
        
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/krishnap25/mauve",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/krishnap25/mauve"],
            reference_urls=[
                "https://arxiv.org/abs/2102.01454",
                "https://github.com/krishnap25/mauve",
            ],
            license=_LICENSE,
        )
    def compute_mauve(
        self,
        predictions,
        references,
    ):
        scores = []
        for pred, ref in zip(predictions, references):
            out = mauve.compute_mauve(
                p_text=predictions,
                q_text=references,
                p_features=self.p_features,
                q_features=self.q_features,
                p_tokens=self.p_tokens,
                q_tokens=self.q_tokens,
                num_buckets=self.num_buckets,
                pca_max_data=self.pca_max_data,
                kmeans_explained_var=self.kmeans_explained_var,
                kmeans_num_redo=self.kmeans_num_redo,
                kmeans_max_iter=self.kmeans_max_iter,
                featurize_model_name=self.featurize_model_name,
                device_id=self.device_id,
                max_text_length=self.max_text_length,
                divergence_curve_discretization_size=self.divergence_curve_discretization_size,
                mauve_scaling_factor=self.mauve_scaling_factor,
                verbose=self.verbose,
                seed=self._seed,
            )
            scores.append(out.mauve)
        return scores


    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        scores = self.compute_mauve(predictions=predictions, references=references)
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
            reduced_scores.append(reduce_fn(self.compute_mauve(extended_preds, refs)))
        return {
            "score":  float(np.mean(reduced_scores)), "reduced_scores" : reduced_scores
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
             "score":  float(np.mean(reduced_scores)), "reduced_scores" : reduced_scores
        }