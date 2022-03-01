import os
import subprocess
import sys
import zipfile
from collections import defaultdict

import gdown
import nltk
import numpy as np
import wget

import utils
# file metrics import
#
from vars import *


class Blanche:
    def __init__(self):
        self.metrics = ["bleu", "rouge", "meteor", "bleurt", "bert_score", "questeval", "BARTScore"]
        self.dep_solved = {}
        self.test_set = {"id": "", "references": "", "predictions": "", "check": False}  # path to test set
        self.init_dep_solved()

    def update_test_set(self, id_test_set, labels, predictions):
        if not os.path.exists(labels) and os.path.exists(predictions):
            print("One or both paths don't exist!")
            return False
        self.test_set["id"] = id_test_set
        self.test_set["references"] = PATH_D + "/" + labels
        self.test_set["predictions"] = PATH_D + "/" + predictions
        self.test_set["check"] = True

        print("Test set updated!")
        return True

    def init_dep_solved(self):
        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])  # get all python packages
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        print(installed_packages)
        for metric in self.metrics:
            print(metric)
            self.dep_solved.update({metric: []})
            # set for all metrics python dependencies that must be installed for use it
            pth = metric + "/" + PATH_DEP
            dep_list = []
            if os.path.exists(pth):
                filed = open(pth, "r")
                content = filed.read()
                dep_list = content.split("\n")
                filed.close()
            removed_index = []
            if dep_list:
                for i in range(0, len(dep_list)):
                    if dep_list[i] in installed_packages:
                        removed_index.append(i)  # filter list of dependency if already installed
                filtered_list = [j for i, j in enumerate(dep_list) if i not in removed_index]

                self.dep_solved[metric] = filtered_list  # save the dependencies that must be installed

    def load(self, metric):
        dependencies = self.dep_solved[metric]
        if dependencies:  # if there's any dependencies when we load a metrics, install it!
            for dep in dependencies:
                script = "pip install " + dep
                os.system(script)
            self.dep_solved[metric] = []
        if metric == "meteor":
            NLTK_VERSION = nltk.__version__
            nltk.download('wordnet')
            if NLTK_VERSION >= "3.6.4":
                nltk.download("punkt")
        if metric == "bleurt":
            if not os.path.isdir(BLEURT_MODEL_PATH):
                url = "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
                path = "bleurt/BLEURT-20.zip"
                print("init download BLEURT model")
                folder = metric + "/"
                self.download_and_extract(url=url, storage_path=path, folder=folder)

    @staticmethod
    def download_and_extract(url, storage_path, folder, unpackable=True, from_drive=False):
        """
        Parameters
        ---------
        url: file to download
        storage_path: path to save compressed file
        folder: folder to extract tar/zip
        unpackable: if True extract file
        from_drive: file downloaded from drive
        """
        import tarfile
        filename, file_extension = os.path.splitext(storage_path)
        if from_drive:
            gdown.download(url, storage_path, quiet=False)
        else:
            wget.download(url, out=storage_path)
        if unpackable:
            print("\nExtracting compressed model...")
            if file_extension == '.zip':
                with zipfile.ZipFile(storage_path, 'r') as zip_ref:
                    zip_ref.extractall(folder)
            else:
                t_model = tarfile.open(storage_path)
                t_model.extractall(folder)
                t_model.close()
            os.remove(storage_path)

    @staticmethod
    def check_dependencies(r_file):
        import subprocess
        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        filed = open(r_file, "r")
        content = filed.read()
        dep_list = content.split("\n")
        filed.close()
        for dep in dep_list:
            if dep not in installed_packages:
                script = "pip install " + dep
                os.system(script)
        print("Dependencies installed!")

    """
    Bleu metric
    citation:
    @inproceedings{papineni-etal-2002-bleu,
    title = "{B}leu: a Method for Automatic Evaluation of Machine Translation",
    author = "Papineni, Kishore  and Roukos, Salim  and Ward, Todd  and Zhu, Wei-Jing",
    booktitle = "Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2002",
    address = "Philadelphia, Pennsylvania, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P02-1040",
    doi = "10.3115/1073083.1073135",
    pages = "311--318",
    }

    description:
    Bleu is a metric born in 2002 to evaluate translation generate texts.
    It's an n-gram overlap algorithm between candidate phrase and set of references.
    Scores are calculated by counting each n-gram which occur both in candidate and references translations,then 
    divides by the total number of words in the candidate translation.
    This type of computation is called precision.
    Bleu adopts a modified precision to prevent the high score given to the over-generation of "reasonable" words.
    It's calculated for individual translated segments—generally sentences—by.
    To extends this precison score to all corpus Bleu adds the n-gram counts for all the candidate sentences and
    divide by the number of candidate n-grams in the test corpus.

    task: summarization
    
    relavants aspects captured: adequancy and fluency
    """
    @staticmethod
    def run_bleu(self, references=None, candidates=None):
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        m_name = self.metrics[0]
        self.load(m_name)
        import bleu.bleu_metric as bm
        ref_tokens = []
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        for ref in references:
            ref_tokens.append([ref.split()])
        pred_tokens = []
        for pred in candidates:
            pred_tokens.append(pred.split())
        # prefect match
        file_name = m_name + "/" + m_name + "_" + self.test_set["id"]
        score_list = []
        with open(file_name, "a") as f:
            for r, p in zip(ref_tokens, pred_tokens):
                results = bm.compute_bleu(reference_corpus=r, translation_corpus=p)
                (bleu, precisions, bp, ratio, translation_length, reference_length) = results
                score_list.append(bleu)
                f.write(str(bleu) + "\n")
        return np.mean(score_list)

    @staticmethod
    def test_bleu():
        refs = utils.load_preds("egv_bart_targets.txt")
        cands = utils.load_preds("egv_bart_preds.txt")
        from nltk.translate.bleu_score import sentence_bleu
        with open("bleu/results_barte.txt", "a") as f:
            for r, c in zip(refs, cands):
                f.write(str(sentence_bleu([r.split()], c.split(), weights=(1, 0, 0, 0))) + '\n')

        return sentence_bleu("data/targets_egv_paper.txt", "data/preds_egv_paper.txt")

    """
    Rouge metric
    citation:
    @inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
    }

    description:
    Rouge-N(N stands for n-gram length) is a n-gram overlap metric recall based which counts the maximum number of 
    n-gram co-occuring in candidate summary and a set of reference summaries.
    Unlike Bleu, the score is calculated dividing all co-occurence by number of reference n-gram.
    ROUGE-L(Longest Common Subsequence): LCS-based Fmeasure to estimate the similarity between two summaries (X,Y),
    where precision and recall are determined by dividing the longest subsequence in common LCS(X,Y) with length of
    candidate summary(precision) and reference summary(recall).

    task: summarization
    relavants aspects captured: adequancy and fluency

    Copyright 2020 The HuggingFace Datasets Authors.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
    """
    def run_rouge(self, references=None, candidates=None, rouge_types=None, stemmer_enable=False,
                  use_aggregator=True):
        """
        Parameters
        ----------
        references: list of references for each candidate
        candidates: list of texts to evaluate
        rouge_types: rouge1,rouge2, rougeL
        stemmer_enable: Bool indicating whether Porter stemmer should be used to strip word suffixes
        use_aggregator: return aggregate scores if True
        """
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        m_name = self.metrics[1]
        self.load(m_name)
        import rouge.rouge_scorer as rg
        import rouge.scoring
        file_name = m_name + "/" + m_name + "_" + self.test_set["id"]
        rouge_types = rouge_types
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        aggregator = None
        scores = None
        scorer = rg.RougeScorer(rouge_types=rouge_types, use_stemmer=stemmer_enable)
        if use_aggregator:
            aggregator = rouge.scoring.BootstrapAggregator()
        else:
            scores = []
        for ref, pred in zip(references, candidates):
            score = scorer.score(ref, pred)

            # with open("rouge/barteL.txt","a") as f:
            #     f.write(str(score["rougeL"].fmeasure) + "\n")

            if use_aggregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)
        if use_aggregator:
            result = aggregator.aggregate()
            print(result)
        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key] for score in scores)
        if use_aggregator:
            for rouge_type in rouge_types:
                score = result[rouge_type].mid.fmeasure
                with open(file_name, "a") as f:
                    f.write("[" + rouge_type + "] = " + str(score) + "\n")
        return result

    def run_meteor(self, references=None, candidates=None, alpha=0.9, beta=3, gamma=0.5):
        """
        Parameters
        ----------
        references: ...
        candidates: ...
        alpha: Parameter for controlling relative weights of precision and recall. default: 0.9
        beta: Parameter for controlling shape of penalty as a function of fragmentation. default 3
        gamma: Relative weight assigned to fragmentation penalty. default 0.5
        
        Return
        ------
        Arithmetic mean of scores computed
        """
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        m_name = self.metrics[4]
        self.load(m_name)
        file_name = m_name + "/" + m_name + "_" + self.test_set["id"]
        from nltk.translate import meteor_score
        NLTK_VERSION = nltk.__version__
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        if NLTK_VERSION >= "3.6.4":
            from nltk import word_tokenize
            scores = [
                meteor_score.single_meteor_score(
                    word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
                )
                for ref, pred in zip(references, candidates)
            ]
        else:
            scores = [
                meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
                for ref, pred in zip(references, candidates)
            ]
        with open(file_name, "a") as f:
            for element in scores:
                f.write(str(element) + "\n")
        return {"meteor": np.mean(scores)}

    """
    Bleurt metric
    citation:
    @misc{sellam2020bleurt,
    title={BLEURT: Learning Robust Metrics for Text Generation}, 
    author={Thibault Sellam and Dipanjan Das and Ankur P. Parikh},
    year={2020},
    eprint={2004.04696},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
    }
    
    description:
    BLEURT is a text-generation metric based on BERT which is pre-trained on artificial dataset made up of random
    perturbated Wikipedia sentences augmented with a diverse set of lexical and semantic-level supervision signals.
    Then BERT is fine-tuned on WMT Metric Shared Task.

    task: summarization, machine translation
    """
    def run_bleurt(self, references=None, candidates=None):
        """
        Parameters
        ----------
        references: list of sources text for candidate
        candidates: list of texts to evaluate
        """
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        m_name = self.metrics[5]
        self.load(self.metrics[5])
        file_name = m_name + "/" + m_name + "_" + self.test_set["id"]
        # checkpoint="bleurt/checkpoint/bleurt-base-128.zip"
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        import bleurt.score as bs
        scorer = bs.BleurtScorer(BLEURT_MODEL_PATH)
        # i=0
        all_scores = []
        with open(file_name, "a") as f:
            for i in range(0, len(references)):
                scores = scorer.score(references=[references[i]], candidates=[candidates[i]])
                all_scores.append(scores)
                f.write(str(scores) + "\n")
                assert type(scores) == list and len(scores) == 1
                print(i)
        return np.mean(all_scores)

    """ 
    BARTScore metric
    citation:
    @misc{yuan2021bartscore,
    title={BARTScore: Evaluating Generated Text as Text Generation}, 
    author={Weizhe Yuan and Graham Neubig and Pengfei Liu},
    year={2021},
    eprint={2106.11520},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
    }
    task: summarization, machine translation
    description:
    BARTScore argue for a formulation of evaluation of generated text as a text generation problem, directly
    evaluating text through the lens of its probability of being generated
    from or generating other textual inputs and outputs.
    This metric uses a pre-trained seq2seq model BART.
    There are no extra parameters beyond those used in pre-training itself, and it is an unsupervised metric that
    doesn’t require human judgments to train.
    BARTSCORE can better support evaluation of generated text from different perspectives (informativeness,
    coherence, factuality) by adjusting the input and the output of the text generation problem.
    BARTSCORE can be further enhanced by (i) providing textual prompts that bring the evaluation task closer to the
    pre-training task, orpdating the underlying model by fine-tuning BART based on downstream generation task.
    Faithfulness (s → h): from source document to hypothesis p(h|s, θ)
    Precision (r → h): from reference text to system-generated text p(h|r, θ)
    Recall (h → r): from system-generated text to reference text p(r|h, θ)
    F score (r ↔ h): Consider both directions and use the arithmetic average of Precision and Recall ones
    """
    def run_BARTScore(self, references=None, candidates=None, device='cpu'):
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        m_name = self.metrics[9]
        self.load(m_name)
        file_name = m_name + "/" + m_name + "_" + self.metrics["id"]
        import BARTScore.bart_score as bt
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        bart_scorer = bt.BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        score_list = []
        with open(file_name, "a") as f:
            for ref, cand in zip(references, candidates):
                score = np.mean(bart_scorer.score(ref, cand))
                score_list.append(score)
                f.write(str(score) + "\n")
        return np.mean(score_list)

    """
    BERTScore metric
    @misc{zhang2020bertscore,
    title={BERTScore: Evaluating Text Generation with BERT}, 
    author={Tianyi Zhang and Varsha Kishore and Felix Wu and Kilian Q. Weinberger and Yoav Artzi},
    year={2020},
    eprint={1904.09675},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
    }
    description:
    Given a reference sentence x = <hx1, . . . , xki> and a candidate sentence xˆ = <hxˆ1, . . . , xˆli>, BERTScore
    uses a contextual embeddings to represent the tokens, and compute matching using cosine similarity
    The representation for each word piece is computed with a Transformer encoder by BERT model
    FBert: 2 * PBert * RBert / (PBert + RBert) 
    """
    def run_BERTScore(self, references=None, candidates=None):
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        m_name = self.metrics[6]
        self.load(m_name)
        file_name = m_name + "/" + m_name + "_" + self.test_set["id"]
        import bert_score
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        # scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        P, R, F1 = bert_score.score(candidates, references, lang="en", rescale_with_baseline=True, verbose=True)
        with open(file_name, "a") as fl:
            for f in F1:
                fl.write(str(f) + "\n")
        return np.mean(F1)

    """
    QUESTEval metric 
    citation:
    @article{scialom2020QuestEval,
    title={QuestEval: Summarization Asks for Fact-based Evaluation},
    author={Scialom, Thomas and Dray, Paul-Alexis and Gallinari Patrick and Lamprier Sylvain and Piwowarski
    Benjamin and Staiano Jacopo and Wang Alex},
    journal={arXiv preprint arXiv:2103.12693},
    year={2021}
    }
    description:
    QUESTEval is a framework for evaluating summarization system composed of QG and QA components.
    Question Answering: T5 model extracts answer from source document given document and a question to answer
    QA(r|T, q).
    It is crucial for the QA model to be able to predict when a question is unanswerable, QA component thus
    includes the unanswerable token (e).
    Question Generation: T5 model fine-tune to likelihood human questions.
    QUESTEval considers all the named entities and nouns from the source document as answers and qustions are
    generated via beam search
    A summary is deemed inconsistent with respect to its source text if, given a question, the answer differs when
    conditioned on summary or document.
    An answer could be expresses in some way, but F1 score consider only overlap from predicted answer and the
    corrisponding ground truth
    So QUESTEval uses QA confident of answerability (1- QA(e)) in this way the answerability is misured
    indipendently of the way the answer is expressed.
    Query weighter distinguish important question from anecdotal ones: W(q, D) denotes the probability that
    question is important for the document, given a source document
    D, each question q ∈ QG(D) is labeled as important if the corresponding human summary contains the answer.

    Precision: F1 for each answer generate from document and gold answer divided by number of pair question-answer
    set genereted by summary
    Recall: (W(q, D)(1 − QA(e|S, q)))/W(q, D) for each pairs answer-question in source document
    Score: 2*Prec*Rec/(Prec + Rec)

    relevants aspects captured: factuality, text relevance
    """
    def run_questeval(self, references=None, candidates=None, sources=None, task='text2text', do_weighter=False,
                      no_cuda=True):
        """
        Parameters
        ----------
        references: list of references for each hypotesis
        candidates: list of claims to be evaluated
        sources: list of source documents for summarization task
        task: there is many type of task from questeval: summarization, text2text, data2text
        no_cuda: True use cpu, False use gpu, if available
        do_weighter: weight for summarization task
        """
        if sources is None:
            sources = []
        if candidates is None:
            candidates = []
        if references is None:
            references = []
        m_name = self.metrics[7]
        self.load(m_name)
        file_name = m_name + "/" + m_name + "_" + self.test_set["id"]
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        from questeval.questeval_metric import QuestEval
        questeval = QuestEval(task=task, do_weighter=do_weighter, no_cuda=no_cuda)
        score_list = []
        with open(file_name, "a") as f:
            for cand, ref in zip(candidates, references):
                score = questeval.corpus_questeval(hypothesis=cand, list_references=[ref], sources=sources)
                score_list.append(score)
                f.write(str(score) + "\n")
        return np.mean(score_list)

    def length(self):
        candidates = []
        if self.test_set["check"]:
            candidates = utils.load_preds(self.test_set["predictions"])
        counter = 0
        for candidate in candidates:
            counter = counter + len(candidate.split(" "))
        return counter / len(candidates)

    def count_ngrams(self, tokens, n):
        counts = defaultdict(int)
        for ngram in self.ngrams(tokens, n):
            counts[' '.join(ngram)] += 1
        return counts

    def repetitiveness(self):
        candidates = None
        if self.test_set["check"]:  # use test set if exists
            candidates = utils.load_preds(self.test_set["predictions"])
        from collections import Counter
        total_sum = 0
        for candidate in candidates:
            monograms = candidate.split(" ")
            n_words = len(monograms)
            m_counted = Counter(monograms)
            for ngram in m_counted.values():
                if ngram > 1:
                    total_sum = total_sum + 1  # if a word  that repeats itself is found
            total_sum = total_sum + n_words
        print(total_sum / len(candidates))

    def abstractness(self, n=1):
        references = []
        candidates = []
        if self.test_set["check"]:  # use test set if exists
            references = utils.load_preds(self.test_set["references"])
            candidates = utils.load_preds(self.test_set["predictions"])
        total_match = 0
        n_words = 0
        for reference, candidate in zip(references, candidates):
            match = 0
            monograms = candidate.split(" ")
            n_words = n_words + len(monograms)  # count all words in test set
            if n > len(monograms):
                return "Not possible to create " + str(n) + "-grams, too many few words"
            for w2 in self.ngrams(monograms, n):
                substr = " ".join(w2)
                if substr not in reference:
                    match = match + 1
            # n_words=n_words+1 #counter for total n-gram number
            total_match = total_match + match
        return total_match / n_words

    @staticmethod
    def ngrams(tokens, n):  # provides an iterable object of n-gram
        ngram = []
        for token in tokens:
            if len(ngram) < n:
                ngram.append(token)
            else:
                yield ngram
                ngram.pop(0)
                ngram.append(token)
        if len(ngram) == n:
            yield ngram
