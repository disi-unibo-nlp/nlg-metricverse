# coding=utf-8
# Copyright 2021 Open Business Software Solutions, The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Perplexity metric. The part of this file is adapted from Perplexity implementation
of datasets package. See
https://github.com/huggingface/datasets/blob/main/metrics/perplexity/perplexity.py """

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable, Dict
import datasets
from datasets import logging
from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import requirement_message


_LICENSE= """ """
_DESCRIPTION = """ """
_CITATION = """ """



_KWARGS_DESCRIPTION = """

"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PerplexityPlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            model_id = 'gpt2',
            add_start_token: bool = False,
            device = "cpu",
            **kwargs,
    ):
        self.model_id = model_id
        self.add_start_token = add_start_token
        self.device = device
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of Perplexity score from the implementation
        of Perplexity computation. The code is sourced from a specific
        commit on the master branch, in order to keep things stable. See
        https://github.com/huggingface/datasets/blob/main/metrics/perplexity/perplexity.py

        """
        if self.device is not None:
            assert self.device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if self.device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.pretrained_model = self.model.to(self.device)
        self.tokenizer =  AutoTokenizer.from_pretrained(self.model_id)
        
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/huggingface/datasets/blob/main/metrics/perplexity",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/huggingface/datasets/blob/main/metrics/perplexity"],
            reference_urls=[
                "https://github.com/huggingface/datasets/blob/main/metrics/perplexity/",
                "https://huggingface.co/docs/transformers/perplexity",
            ],
            license=_LICENSE,
        )
    def perplexity_score(self, predictions: EvaluationInstance, batch_size: int = 16):
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if self.add_start_token:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = self.pretrained_model.config.max_length - 1
        else:
            max_tokenized_len = self.pretrained_model.config.max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if self.add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if self.add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.pretrained_model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp2(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            batch_size: int = 16,
            **kwargs,
    ):
        scores = self.perplexity_score(predictions=predictions, batch_size=batch_size)
        return scores

    def _compute_single_pred_multi_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            batch_size: int = 16,
            reduce_fn: Callable = None,
            **kwargs,
    ):  
        scores = self.perplexity_score(predictions=predictions, batch_size=batch_size)
        return scores

    def _compute_multi_pred_multi_ref(
        self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            batch_size: int = 16,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        
        inputList = []
        for prediction in predictions:
            inputList += prediction
        scores = self.perplexity_score(predictions=inputList, batch_size=batch_size)
        return scores