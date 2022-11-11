import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from pyemd import emd_with_flow

if os.environ.get('MOVERSCORE_MODEL'):
    model_name = os.environ.get('MOVERSCORE_MODEL')
else:
    model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
model.eval()


def n_gram_distance_visualization(reference, translation, is_flow=True, device='cuda:0'):
    if not isinstance(reference, str) and not isinstance(translation, str):
        raise ValueError("'reference' (and 'translation') must be a single string.")
    else:
        idf_dict_ref = defaultdict(lambda: 1.)
        idf_dict_hyp = defaultdict(lambda: 1.)

        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding([reference], model, tokenizer,
                                                                                     idf_dict_ref)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding([translation], model, tokenizer,
                                                                                     idf_dict_hyp)

        ref_embedding = ref_embedding[-1]
        hyp_embedding = hyp_embedding[-1]

        raw = torch.cat([ref_embedding, hyp_embedding], 1)
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)

        distance_matrix = batched_cdist_l2(raw, raw)
        masks = torch.cat([ref_masks, hyp_masks], 1)
        masks = torch.einsum('bi,bj->bij', (masks, masks))
        distance_matrix = masks * distance_matrix

        i = 0
        c1 = np.zeros(raw.shape[1], dtype=np.float)
        c2 = np.zeros(raw.shape[1], dtype=np.float)
        c1[:len(ref_idf[i])] = ref_idf[i]
        c2[len(ref_idf[i]):] = hyp_idf[i]

        c1 = _safe_divide(c1, np.sum(c1))
        c2 = _safe_divide(c2, np.sum(c2))

        dst = distance_matrix[i].double().cpu().numpy()

        if is_flow:
            _, flow = emd_with_flow(c1, c2, dst)
            new_flow = np.array(flow, dtype=np.float32)
            res = new_flow[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i]) + len(hyp_tokens[i]))]
        else:
            res = 1. / (1. + dst[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i]) + len(hyp_tokens[i]))])

        r_tokens = ref_tokens[i]
        h_tokens = hyp_tokens[i]

        fig, ax = plt.subplots(figsize=(len(r_tokens) * 0.8, len(h_tokens) * 0.8))
        im = ax.imshow(res, cmap='Blues')

        ax.set_xticks(np.arange(len(h_tokens)))
        ax.set_yticks(np.arange(len(r_tokens)))

        ax.set_xticklabels(h_tokens, fontsize=10)
        ax.set_yticklabels(r_tokens, fontsize=10)
        plt.xlabel("System Translation", fontsize=14)
        plt.ylabel("Human Reference", fontsize=14)
        plt.title("Flow Matrix", fontsize=14)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        #    for i in range(len(r_tokens)):
        #        for j in range(len(h_tokens)):
        #            text = ax.text(j, i, '{:.2f}'.format(res[i, j].item()),
        #                           ha="center", va="center", color="k" if res[i, j].item() < 0.6 else "w")
        fig.tight_layout()
        # plt.show()


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1):
    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                              tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                              idf_dict)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i + batch_size],
                                          attention_mask=mask[i:i + batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)


def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res


def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        result = model(x, attention_mask=attention_mask)
    if model_name == 'distilbert-base-uncased':
        return result[1]
    else:
        return result[2]


def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]"):
    tokens = [["[CLS]"] + truncate(tokenize(a)) + ["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    return padded, padded_idf, lens, mask, tokens


def truncate(tokens):
    if len(tokens) > tokenizer.model_max_length - 2:
        tokens = tokens[0:(tokenizer.model_max_length - 2)]
    return tokens


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask