# coding=utf-8
import datasets
from moverscore_v2 import get_idf_dict, word_mover_score
import numpy as np

from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import requirement_message

_CITATION = """\
@inproceedings{zhao2019moverscore,
  title = {MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  month = {August},
  year = {2019},
  author = {Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
"""

_DESCRIPTION = """\
MoverScore (Zhao et.al, 2019) is an automated evaluation metric assigning a single holistic score to any
system-generated text (neural or non-neural) by comparing it against human references for semantic content
matching. It is a monolingual measure evluating meaning similarities between pairs of sentences written in
the same language. It combines contextualized representations coming from language models (trained to capture
distant semantic dependencies) with the Word Mover's distance (WMD). So, MoverScore generalizes WMD by working
on n-grams. Specifically, it computes the minimum cost of transforming (transportation distance) the generated
text to the reference text, taking into account Euclidean distance between vector representations of n-gram as
well as their document frequencies. According to the authors, MoverScore can be seen as a generalization of
BertScore. Both of them use contextualized representations, but they have a different focus. BertScore aligns
each hypothesis word with a single reference word (1:1), while MoverScore makes a soft alignment (1:N).
MoverScore demonstrates strong generalization capability across multiple tasks, achieving much higher correlation
with human judgments than BLEU on machine translation, summarization and image captioning.

BOUNDS
Intuitively, the metric assigns a perfect score to the system text if it conveys the same meaning as the
reference text. Any deviation from the reference content can then lead to a reduced score, e.g., the
system text contains more (or less) content than the reference, or the system produces ill-formed text
that fails to deliver the intended meaning. In general, higher scores refer to better performance.

WEAKNESSES
The paradigm of reference-based measures is useful for targeted generation tasks such as translation and
summarization where matching a set of references is paramount. It is, however, unsuitable for open-ended
generation where there typically are several plausible continuations for each context and creative
generations are desirable.

PROPERTY
IDF-weighted n-gram soft-alignment via ELMo/BERT contextualized embeddings

CATEGORY
unsupervised; embedding-based

TASKS
MT, SUM, D2T, IC

References:
[Official repository](https://github.com/AIPHES/emnlp19-moverscore).
"""

_KWARGS_DESCRIPTION = """
Computes MoverScore.
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    version: which MoverScore version use for computing scores.
        Choose between "1" or "2".
        Version 1 is the implementation presented in the paper, mantained for reproducibility but slow
        to run. Version 2 runs faster by disabling powermean but is a bit worse in performance.
        Default: 2.
Returns:
    'score': mover score.
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MoverscorePlanet(MetricForLanguageGeneration):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/AIPHES/emnlp19-moverscore"],
            reference_urls=[
                "https://github.com/AIPHES/emnlp19-moverscore"
            ],
        )

    def _download_and_prepare(self, dl_manager):
        try:
            import moverscore
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="moverscore", package_name="moverscore"))

    def _compute_single_pred_single_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            **kwargs
    ):
        idf_dict_hyp = get_idf_dict(predictions)  # idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = get_idf_dict(references)  # idf_dict_ref = defaultdict(lambda: 1.)

        scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp,
                                  stop_words=[], n_gram=1, remove_subwords=True)
        return {"score": scores}

    def _compute_single_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            **kwargs
    ):
        res = []
        for reference in references:
            idf_dict_hyp = get_idf_dict(predictions)  # idf_dict_hyp = defaultdict(lambda: 1.)
            idf_dict_ref = get_idf_dict(reference)  # idf_dict_ref = defaultdict(lambda: 1.)

            score = word_mover_score(reference, predictions, idf_dict_ref, idf_dict_hyp,
                                     stop_words=[], n_gram=1, remove_subwords=True)
            res.append(score)
        scores = np.mean(res)
        return {"score": scores}

    def _compute_multi_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            **kwargs
    ):
        res = []
        for prediction in predictions:
            score = self._compute_single_pred_multi_ref(
                predictions=prediction, references=references, reduce_fn=reduce_fn)
            res.append(score["score"])
        scores = np.mean(res)
        return {"score": scores}
