import numpy as np
import pandas as pd
import pickle as pkl
import os
import torch
import requests
import tarfile

from tqdm.auto import tqdm
from collections import defaultdict
from scipy.stats import pearsonr

import bert_score

from nlgmetricverse.utils.sys import log

wmt17_sys_to_lang_pairs = ['cs-en', 'de-en', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en']
wmt17_sys_from_lang_pairs = ['en-cs', 'en-de', 'en-lv', 'en-ru', 'en-tr', 'en-zh']
wmt17_sys_lang_pairs = wmt17_sys_to_lang_pairs + wmt17_sys_from_lang_pairs


def get_wmt17_sys_data(lang_pair):
    first, second = lang_pair.split("-")

    human_scores = pd.read_csv(
        "wmt17/manual-evaluation/DA-syslevel.csv", delimiter=" ")

    with open("wmt17/input/wmt17-metrics-task/"
              "wmt17-submitted-data/txt/references/newstest2017-{}{}-ref.{}".format(first, second, second),
              encoding = "utf-8") as f:
        refs = f.read().strip().split("\n")

    gold_dict = dict(zip(human_scores[human_scores['LP'] == lang_pair]['SYSTEM'],
                         human_scores[human_scores['LP'] == lang_pair]['HUMAN']))
    gold_scores = []

    lang_dir = "wmt17/input/" \
               "wmt17-metrics-task/wmt17-submitted-data/" \
               "txt/system-outputs/newstest2017/{}".format(lang_pair)
    systems = [system[13:-6] for system in os.listdir(lang_dir)]

    refs *= len(systems)
    cands = []

    for system in systems:
        with open(os.path.join(lang_dir, "newstest2017.{}.{}".format(system, lang_pair)), encoding="utf-8") as f:
            cand_sys = f.read().strip().split("\n")
        gold_scores.append(gold_dict[system])

        cands += cand_sys
    return refs, cands, gold_scores, systems


def get_wmt17_sys_bert_score(lang_pair, scorer, cache=False, from_en=True, batch_size=64):
    filename = ''
    if from_en:
        if scorer.idf:
            filename = "cache_score/from_en/17/{}/wmt17_seg_from_{}_{}_idf.pkl".format(scorer.model_type,
                                                                                       *lang_pair.split('-'))
        else:
            filename = "cache_score/from_en/17/{}/wmt17_seg_from_{}_{}.pkl".format(scorer.model_type,
                                                                                   *lang_pair.split('-'))
    else:
        if scorer.idf:
            filename = "cache_score/to_en/17/{}/wmt17_seg_to_{}_{}_idf.pkl".format(scorer.model_type,
                                                                                   *lang_pair.split('-'))
        else:
            filename = "cache_score/to_en/17/{}/wmt17_seg_to_{}_{}.pkl".format(scorer.model_type, *lang_pair.split('-'))

    if os.path.exists(filename):
        with open(filename, "rb", encoding="utf-8") as f:
            return pkl.load(f)
    else:
        refs, cands, gold_scores, systems = get_wmt17_sys_data(lang_pair)
        if scorer.idf:
            scorer.compute_idf(refs)
        raw_scores = scorer.score(cands, refs, batch_size=batch_size)
        scores = [s.view(len(systems), -1).mean(dim=-1) for s in raw_scores]

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb", encoding="utf-8") as f:
            pkl.dump((scores, gold_scores), f)

    return scores, gold_scores


def get_wmt17_sys_results(
        model=None,
        log_file="wmt17_log.csv",
        idf=False,
        batch_size=64,
        lang_pairs=None
):

    if model is None:
        model = ["roberta-large"]
    if lang_pairs is None:
        lang_pairs = wmt17_sys_from_lang_pairs
    torch.set_grad_enabled(False)

    header = 'model_type'
    for lang_pair in lang_pairs + ['avg']:
        header += f',{lang_pair}'
    log(header)
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding="utf-8") as f:
            log(header, file=f)

    log(model)
    for model_type in model:
        scorer = bert_score.scorer.BERTScorer(model_type=model_type, idf=idf)
        results = defaultdict(dict)
        for lang_pair in tqdm(lang_pairs):
            scores, gold_scores = get_wmt17_sys_bert_score(lang_pair, scorer, batch_size=batch_size, cache=True,
                                                           from_en=False)
            for s, name in zip(scores, ["P", "R", "F"]):
                results[lang_pair][f"{model_type} {name}"] = np.mean(pearsonr(gold_scores, s)[0])

        for name in ["P", "R", "F"]:
            temp = []
            for lang_pair in lang_pairs:
                temp.append(results[lang_pair][f"{model_type} {name}"])
            results["avg"][f"{model_type} {name}"] = np.mean(temp)

            msg = f"{model_type} {name} (idf)" if idf else f"{model_type} {name}"
            for lang_pair in lang_pairs + ['avg']:
                msg += f",{results[lang_pair][f'{model_type} {name}']}"
            log(msg)
            with open(log_file, "a", encoding="utf-8") as f:
                log(msg, file=f)

        del scorer


def wmt17_download_data():
    starting_dir = os.getcwd()
    directory = "wmt17"
    parent_dir = os.path.curdir
    path = os.path.join(parent_dir, directory)
    if not os.path.isdir('./wmt17'):
        directory = "wmt17"
        parent_dir = os.path.curdir
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    os.chdir(path)
    if not os.path.isfile('./wmt17.tgz'):
        url = 'http://ufallab.ms.mff.cuni.cz/~bojar/wmt17-metrics-task-package.tgz'
        r = requests.get(url, allow_redirects=True)
        open('wmt17.tgz', 'wb', encoding="utf-8").write(r.content)
    if not os.path.isdir('input'):
        tar = tarfile.open("wmt17.tgz", encoding="utf-8")
        tar.extractall()
        tar.close()
    directory = "input"
    parent_dir = os.path.curdir
    path = os.path.join(parent_dir, directory)
    os.chdir(path)
    if not os.path.isdir('wmt17-metrics-task'):
        tar = tarfile.open("wmt17-metrics-task.tgz", encoding="utf-8")
        tar.extractall()
        tar.close()
    os.chdir(starting_dir)

