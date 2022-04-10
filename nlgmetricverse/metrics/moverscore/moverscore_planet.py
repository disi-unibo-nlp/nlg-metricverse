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
"""

_KWARGS_DESCRIPTION = """
Computes MoverScore.
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    version: which version to compute scoring.
        Choose between "1" or "2".
        Second one is more recent and faster.
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
