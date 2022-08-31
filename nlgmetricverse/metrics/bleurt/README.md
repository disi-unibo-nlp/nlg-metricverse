# Metric Card for BLEURT

## Metric Description
Bilingual Evaluation Understudy with Representations from Transformers (BLEURT) is a fully learned evaluation metric modeling human judgments for generated text, i.e., it is a regression model trained on ratings data.
It takes a pair of sentences as input, a reference and a candidate, and it returns a score that indicates to what extent the candidate is fluent and conveys the mearning of the reference.
BLEURT is based on [`BERT`](https://arxiv.org/abs/1810.04805) and a novel (additional) pre-training scheme based on millions of synthetic reference-candidate pairs, generated through perturbations (i.e., mask-filling, backtranslation, dropping words) and aimed to help the model generalize (greater robustness).
Differently from existing sentence pairs datasets, synthetic data allow to capture the errors and alterations that NLG systems produce (e.g., omissions, repetitions, nonsensical substitutions).
Extra BERT pre-training on such syntethic data considers several lexical- and semantic-level supervision signals with a multitask loss, i.e., a weighted sum aggregation of task-level regression or classification losses (BLEU/ROUGE/BERTScore emulation, backtranslation likelihood/flag, textual entailment).
So, BLEURT models are trained in three steps: regular BERT pre-training (Devlin et al., 2019), pre-training on synthetic data, and fine-tuning on task-specific ratings (like translation and/or data-to-text using public WMT human annotations).
Note: rating data prediction at the third step is done with a classification layer on top of BERT's [CLS].

You may run BLEURT out-of-the-box or fine-tune it for your specific application (the latter is expected to perform better).
BLEURT is comparable to [`sentence-BLEU`](https://en.wikipedia.org/wiki/BLEU), [`BERTscore`](https://arxiv.org/abs/1904.09675), and [`COMET`](https://github.com/Unbabel/COMET).
This implementation follows the original one: it uses Tensorflow, and it benefits greatly from modern GPUs (it runs on CPU too).

### Inputs
- **predictions** (`list`): prediction/candidate sentences.
- **references** (`list`): reference sentences.
- **config_name** (`str`): BLEURT checkpoint. Defaults to `BLEURT-20` if None.

### Outputs
BLEURT outputs a dictionary with the following values:
- **score** (`float`): average BLEURT score.
- **scores** (`list`): list of individual BLEURT scores.
- **checkpoint** (`string`): selected BLEURT checkpoint.

## Bounds
Different BLEURT checkpoints yield different scores.
The currently recommended checkpoint [`BLEURT-20`](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip) generates scores which are roughly between 0 and 1 (sometimes less than 0, sometimes more than 1), where 0 indicates a random output and 1 a perfect one.
As with all automatic metrics, BLEURT scores are noisy.
For a robust evaluation of a system's quality, we recommend averaging BLEURT scores across the sentences in a corpus.
See the [WMT Metrics Shared Task](http://statmt.org/wmt21/metrics-task.html) for a comparison of metrics on this aspect.

In principle, BLEURT should measure *adequacy*: most of its training data was collected by the WMT organizers who asked to annotators "How much do you agree that the system output adequately expresses the meaning of the reference?" ([WMT Metrics'18](http://www.statmt.org/wmt18/pdf/WMT078.pdf), [Graham et al., 2015](https://minerva-access.unimelb.edu.au/bitstream/handle/11343/56463/Graham_Can-machine-translation.pdf)).
In practice however, the answers tend to be very correlated with *fluency* ("Is the text fluent English?"), and we added synthetic noise in the training set which makes the distinction between adequacy and fluency somewhat fuzzy.

## Examples
```python
# Example with BLEURT-tiny checkpoint, which is very light but also very inaccurate
scorer = NLGMetricverse(metrics=load_metric(
    base_path + "bleurt",
    config_name="bleurt-tiny-128"))
predictions = [
  ["the cat is on the mat", "There is cat playing on the mat"],
  ["Look! what a wonderful day, today.", "Today is a very wonderful day"]
]
references = [
  ["the cat is playing on the mat.", "The cat plays on the mat."], 
  ["Today is a wonderful day", "The weather outside is wonderful."]
]
scores = scorer(predictions=predictions, references=references)
print(scores)
{'total_items': 2, 'empty_items': 0, 'bleurt': {'score': 0.6418270468711853, 'scores': [0.47344332933425903, 0.8102107644081116], 'checkpoint': 'bleurt-tiny-128'}}
```

## Citation
```bibtex
@inproceedings{bleurt,
  title={BLEURT: Learning Robust Metrics for Text Generation},
  author={Thibault Sellam and Dipanjan Das and Ankur P. Parikh},
  booktitle={ACL},
  year={2020},
  url={https://arxiv.org/abs/2004.04696}
}
```

## Further References
- See the project's [README](https://github.com/google-research/bleurt#readme) for more information.
- An overview of BLEURT can be found in the official [Google blog post](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html).
- Further details are provided in the ACL [BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696) and [EMNLP](http://arxiv.org/abs/2110.06341) papers.
