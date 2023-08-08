# Metric Card for CER

## Metric description
Character error rate (CER) is a common metric of the performance of an automatic speech recognition (ASR) system. CER is similar to Word Error Rate (WER), but operates on character instead of word. 

Character error rate can be computed as: 

`CER = (S + D + I) / N = (S + D + I) / (S + D + C)`

where

`S` is the number of substitutions, 

`D` is the number of deletions, 

`I` is the number of insertions, 

`C` is the number of correct characters, 

`N` is the number of characters in the reference (`N=S+D+C`). 

### Inputs
-  **predictions** (`list`): list of transcribtions to score.
-  **references** (`list`): list of references for each speech input.
-  **concatenate_texts** (`boolean`): Whether or not to concatenate sentences before evaluation, set to True for more accurate result.

### Output
-  **cer_score** (`float`): Character error rate. This metric outputs a float representing the character error rate.The **lower** the 
                            CER value, the **better** the performance of the ASR system, with a CER of 0 being a perfect score. 

However, CER's output is not always a number between 0 and 1, in particular when there is a high number of insertions (see [Examples](##Examples) below).

### Results from popular papers
This metric is highly dependent on the content and quality of the dataset, and therefore users can expect very different values for the same model but on different datasets.

Multilingual datasets such as [Common Voice](https://huggingface.co/datasets/common_voice) report different CERs depending on the language, ranging from 0.02-0.03 for languages such as French and Italian, to 0.05-0.07 for English (see [here](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice/ASR/CTC) for more values).

## Bounds
The CER score can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0,1]}##gh-light-mode-only">.

## Examples 
Perfect match between prediction and reference:

```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("cer"))
predictions = ["hello world", "good night moon"]
references = ["hello world", "good night moon"]
scores = scorer(predictions=predictions, references=references)
print(scores)
{
    "cer": {
        'score': 0.0
    }
}
```

Partial match between prediction and reference:

```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("cer"))
predictions = ["this is the prediction", "there is an other sample"]
references = ["this is the reference", "there is another one"]
scores = scorer(predictions=predictions, references=references)
print(scores)
{
    "cer": {
        'score': 0.34146341463414637
    }
}
```

No match between prediction and reference:

```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("cer"))
predictions = ["hello"]
references = ["gracias"]
scores = scorer(predictions=predictions, references=references)
print(scores)
{
    "cer": {
        'score': 1.0
    }
}
```

CER above 1 due to insertion errors:

```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("cer"))
predictions = ["hello world"]
references = ["hello"]
scores = scorer(predictions=predictions, references=references)
print(scores)
{
    "cer": {
        'score':1.2
    }
}
```

## Limitations and bias
CER is useful for comparing different models for tasks such as automatic speech recognition (ASR) and optic character recognition (OCR), especially for multilingual datasets where WER is not suitable given the diversity of languages. However, CER provides no details on the nature of translation errors and further work is therefore required to identify the main source(s) of error and to focus any research effort.

Also, in some cases, instead of reporting the raw CER, a normalized CER is reported where the number of mistakes is divided by the sum of the number of edit operations (`I` + `S` + `D`) and `C` (the number of correct characters), which results in CER values that fall within the range of 0–100%.


## Citation
```bibtex
@inproceedings{morris2004,
author = {Morris, Andrew and Maier, Viktoria and Green, Phil},
year = {2004},
month = {01},
pages = {},
title = {From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.}
}
```

## Further References 
- [Hugging Face Tasks -- Automatic Speech Recognition](https://huggingface.co/tasks/automatic-speech-recognition)