import numpy as np
import pathlib

from utils import common, vars
from metrics.bleu import bleu


def run_bleu(references=None, candidates=None):
    if references is None:
        references = []
    if candidates is None:
        candidates = []
    m_name = vars.METRICS[0]
    common.load(m_name)
    ref_tokens = []
    if vars.data["check"]:  # use test set if exists
        references = common.load_preds(vars.data["references"])
        candidates = common.load_preds(vars.data["predictions"])
    for ref in references:
        ref_tokens.append([ref.split()])
    pred_tokens = []
    for pred in candidates:
        pred_tokens.append(pred.split())
    # prefect match
    current_folder = str(pathlib.Path(__file__).parent.resolve())
    file_name = current_folder + "/" + m_name + "_" + vars.data["id"]
    score_list = []
    with open(file_name, "a") as f:
        for r, p in zip(ref_tokens, pred_tokens):
            results = bleu.compute_bleu(reference_corpus=r, translation_corpus=p)
            (bleu_score, precisions, bp, ratio, translation_length, reference_length) = results
            score_list.append(bleu_score)
            res = "\n" + "bleu_score: " + str(bleu_score) + "\n" +\
                "precisions: " + str(precisions) + "\n" +\
                "bp: " + str(bp) + "\n" +\
                "ratio: " + str(ratio) + "\n" +\
                "translation_length: " + str(translation_length) + "\n" +\
                "reference_length: " + str(reference_length) + "\n"
            f.write(res)
        print(res)
    return np.mean(score_list)
