""" CiDEr metric. The part of this file is adapted from vrama91 implementation.
The follow implementation supports python 3.x version. See
https://github.com/vrama91/cider/blob/master/pyciderevalcap/cider/cider_scorer.py for the original version """


import evaluate
from collections import defaultdict
import numpy as np
import math
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
class CiderPlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            test=None,
            refs=None,
            n=4,
            sigma=6.0,
            **kwargs,
    ):
        self.n = n
        self.sigma = sigma
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)
        
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/vrama91/cider",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/vrama91/cider/blob/master/pyciderevalcap/cider/cider_scorer.py"],
            reference_urls=[
                "https://github.com/vrama91/cider",
                "https://arxiv.org/abs/1411.5726",
            ],
            license=_LICENSE,
        )

    def prepare(self, tests, refs):
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        for test, ref in zip(tests, refs):
            self.cook_append(test, ref)
        self.ref_len = None
        self.compute_doc_freq()
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        

    def precook(self, s, n=4, out=False):
        """
        Takes a string as input and returns an object that can be given to
        either cook_refs or cook_test. This is optional: cook_refs and cook_test
        can take string arguments as well.
        :param s: string : sentence to be converted into ngrams
        :param n: int    : number of ngrams for which representation is calculated
        :return: term frequency vector for occuring ngrams
        """
        words = s.split()
        counts = defaultdict(int)
        for k in range(1,n+1):
            for i in range(len(words)-k+1):
                ngram = tuple(words[i:i+k])
                counts[ngram] += 1
        return counts

    def cook_refs(self,refs, n=4): ## lhuang: oracle will call with "average"
        '''Takes a list of reference sentences for a single segment
        and returns an object that encapsulates everything that BLEU
        needs to know about them.
        :param refs: list of string : reference sentences for some image
        :param n: int : number of ngrams for which (ngram) representation is calculated
        :return: result (list of dict)
        '''
        return [self.precook(s=ref, n=n) for ref in refs]

    def cook_test(self,test, n=4):
        '''Takes a test sentence and returns an object that
        encapsulates everything that BLEU needs to know about it.
        :param test: list of string : hypothesis sentence for some image
        :param n: int : number of ngrams for which (ngram) representation is calculated
        :return: result (dict)
        '''
        return self.precook(s=test, n=n, out=True)

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(self.cook_refs(refs))
            if test is not None:
                self.ctest.append(self.cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val
 
        if len(self.crefs) == 1:    
            self.ref_len = 1
        else:
            self.ref_len = np.log(float(len(self.crefs)))
        scores = []

        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def cider_scorer(self, predictions: EvaluationInstance,references: EvaluationInstance):
        self.prepare(predictions, references)
        scores = self.compute_cider()
        return (float(np.mean(scores)), scores)

    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            **kwargs,
    ):
        (score, scores) = self.cider_scorer(predictions, [references])
        return {
            "score": score
        }

    def _compute_single_pred_multi_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            **kwargs,
    ):
        #it can't reduce beacause the result of cider_scorer is the avg of scores
        (score, scores) = self.cider_scorer(predictions, references)
        return {
            "score": score, "scores" : scores
        }

    def _compute_multi_pred_multi_ref(
        self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            **kwargs,
    ):

        new_predictions= []
        new_references = []
        reduced_scores=[]
        preds_lenghts=[]
        for preds, refs in zip(predictions, references):
            preds_lenghts.append(len(preds))
            for pred in preds:
                new_predictions.append(pred)
                new_references.append(refs)
        (score, scores) = self.cider_scorer(new_predictions, new_references)  
        index=0
        for length in preds_lenghts:
            sub_scores=scores[index:length]
            if np.array(sub_scores).any():
                reduced_scores.append(reduce_fn(scores[index:length]))
            else:
                 reduced_scores.append(0.0)
            index = length
        return {
             "score": sum(reduced_scores) / len(reduced_scores), "scores" : reduced_scores
        }