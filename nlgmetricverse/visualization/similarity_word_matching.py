import os
import sys
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from collections import defaultdict

from nlgmetricverse.utils.common import (
    get_tokenizer,
    get_bert_embedding,
    lang2model,
    model2layers,
    sent_encode,
)
from nlgmetricverse.utils.model import get_model


def similarity_word_matching(
    candidate,
    reference,
    model_type=None,
    num_layers=None,
    lang=None,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
    fname="",
):
    """
    BERTScore metric.
    Args:
        - :param: `candidate` (str): a candidate sentence
        - :param: `reference` (str): a reference sentence
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        - :param: `fname` (str): path to save the output plot
    """
    assert isinstance(candidate, str)
    assert isinstance(reference, str)

    assert lang is not None or model_type is not None, "Either lang or model_type should be specified"

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    idf_dict = defaultdict(lambda: 1.0)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    hyp_embedding, masks, padded_idf = get_bert_embedding(
        [candidate], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding, masks, padded_idf = get_bert_embedding(
        [reference], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim.squeeze(0).cpu()

    # remove [CLS] and [SEP] tokens
    r_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, reference)][1:-1]
    h_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, candidate)][1:-1]
    sim = sim[1:-1, 1:-1]

    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv")
        if os.path.isfile(baseline_path):
            baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
            sim = (sim - baselines[2].item()) / (1 - baselines[2].item())
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}", file=sys.stderr,
            )

    fig, ax = plt.subplots(figsize=(len(r_tokens), len(h_tokens)))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(r_tokens)))
    ax.set_yticks(np.arange(len(h_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(r_tokens, fontsize=10)
    ax.set_yticklabels(h_tokens, fontsize=10)
    ax.grid(False)
    plt.xlabel("Reference (tokenized)", fontsize=14)
    plt.ylabel("Candidate (tokenized)", fontsize=14)
    title = "Similarity Matrix"
    if rescale_with_baseline:
        title += " (after Rescaling)"
    plt.title(title, fontsize=14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    fig.colorbar(im, cax=cax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(h_tokens)):
        for j in range(len(r_tokens)):
            text = ax.text(
                j,
                i,
                "{:.3f}".format(sim[i, j].item()),
                ha="center",
                va="center",
                color="k" if sim[i, j].item() < 0.5 else "w",
            )

    fig.tight_layout()
    if fname != "":
        plt.savefig(fname, dpi=100)
        print("Saved figure to file: ", fname)
    # plt.show()
