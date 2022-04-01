# coding=utf-8
from collections import defaultdict
from itertools import zip_longest
from typing import List, Union, Iterable
import numpy as np
import datasets

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
MoverScore (Zhao et.al, 2019) is a monolingual measure of evaluating the similarity between a sentence pair written in 
the same language. It achieves much higher correlation with human judgments than BLEU on machine translation, 
summarization and image captioning.
"""

_KWARGS_DESCRIPTION = """

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
            import pyemd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="moverscore", package_name="pyemd"))
        else:
            super(MoverscorePlanet, self)._download_and_prepare(dl_manager)
        try:
            import transformers
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="moverscore", package_name="transformers"))
        else:
            super(MoverscorePlanet, self)._download_and_prepare(dl_manager)

    def _compute_single_pred_single_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            version=2
    ):
        if isinstance(predictions[0], List):
            res = []
            score = None
            for prediction, reference in zip(predictions, references):
                newRef = [reference]
                newPred = prediction
                while len(reference) != len(prediction):
                    newItem = ''
                    newPred.append(newItem)
                if version == 2:
                    score = self.corpus_score(newPred, newRef, version=2)
                else:
                    score = self.corpus_score(newPred, newRef, version=1)
                res += score
            scores = np.mean(score)
        else:
            if version == 2:
                scores = self.corpus_score(predictions, references, version=2)
            else:
                scores = self.corpus_score(predictions, references, version=1)
        return {"score": scores}

    def _compute_single_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            version=2
    ):
        if isinstance(predictions[0], List):
            res = []
            score = None
            for prediction, reference in zip(predictions, references):
                newRef = [reference]
                newPred = prediction
                while len(reference) != len(prediction):
                    newItem = ''
                    newPred.append(newItem)
                if version == 2:
                    score = self.corpus_score(newPred, newRef, version=2)
                else:
                    score = self.corpus_score(newPred, newRef, version=1)
                res += score
            scores = np.mean(score)
        else:
            if version == 2:
                scores = self.corpus_score(predictions, references, version=2)
            else:
                scores = self.corpus_score(predictions, references, version=1)
        return {"score": scores}

    def _compute_multi_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            version=2
    ):
        if isinstance(predictions[0], List):
            res = []
            score = None
            for prediction, reference in zip(predictions, references):
                newRef = [reference]
                newPred = prediction
                while len(reference) != len(prediction):
                    newItem = ''
                    newPred.append(newItem)
                if version == 2:
                    score = self.corpus_score(newPred, newRef, version=2)
                else:
                    score = self.corpus_score(newPred, newRef, version=1)
                res += score
            scores = np.mean(score)
        else:
            if version == 2:
                scores = self.corpus_score(predictions, references, version=2)
            else:
                scores = self.corpus_score(predictions, references, version=1)
        return {"score": scores}

    def corpus_score(
            self,
            sys_stream: List[str],
            ref_streams: Union[str, List[Iterable[str]]],
            version=2
    ):
        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]
        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]
        fhs = [sys_stream] + ref_streams
        corpus_score = 0
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")
            hypo, *refs = lines
            corpus_score += self.sentence_score(hypo, refs, version, trace=0)
        corpus_score /= len(sys_stream)

        return corpus_score

    @staticmethod
    def sentence_score(
            hypothesis: str,
            references: List[str],
            version=2,
            trace=0,
    ):
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
        hypothesis = [hypothesis] * len(references)
        if version == 2:
            import moverscore_v2 as mv2
            scores = mv2.word_mover_score(
                references,
                hypothesis,
                idf_dict_ref,
                idf_dict_hyp,
                stop_words=[],
                n_gram=1,
                remove_subwords=False)
        else:
            import moverscore as mv
            scores = mv.word_mover_score(
                references,
                hypothesis,
                idf_dict_ref,
                idf_dict_hyp,
                stop_words=[],
                n_gram=1,
                remove_subwords=False)
        sentence_score = np.mean(scores)
        if trace > 0:
            print(hypothesis, references, sentence_score)

        return sentence_score
