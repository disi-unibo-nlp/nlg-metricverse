# Metric Card for BLEU

## Metric Description
BLEU (bilingual evaluation understudy) scores were originally developed in the context of machine translation, but they are applied in other generation tasks as well.
Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU.
BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.
For BLEU scoring, we require a dataset <img src="https://render.githubusercontent.com/render/math?math={Y}##gh-light-mode-only"> consisting of instances <img src="https://render.githubusercontent.com/render/math?math={(a, B)}##gh-light-mode-only"> where <img src="https://render.githubusercontent.com/render/math?math={a}##gh-light-mode-only"> is a candidate (a model prediction) and <img src="https://render.githubusercontent.com/render/math?math={B}##gh-light-mode-only"> is a set of gold texts.

_What percentage of predicted n-grams (text string clusters) can be found in the reference text?_<br>
The metric has two main components.
- Modified n-gram precision. A direct application of precision would divide the number of correct n-grams in the candidate (n-grams that appear in any translation) by the total number of n-grams in the candidate. This has a degenerate solution in which the predicted output contains only one n-gram. BLEU's modified version substitutes the actual count for each n-gram <img src="https://render.githubusercontent.com/render/math?math={s}##gh-light-mode-only"> in the candidate by the maximum number of times <img src="https://render.githubusercontent.com/render/math?math={s}##gh-light-mode-only"> appears in any gold text.
- Brevity penalty (BP). To avoid favoring outputs that are too short, a penalty is applied. Let <img src="https://render.githubusercontent.com/render/math?math={r}##gh-light-mode-only"> be the sum of all minimal absolute length differences between candidates and referents in the dataset <img src="https://render.githubusercontent.com/render/math?math={Y}##gh-light-mode-only">, and let <img src="https://render.githubusercontent.com/render/math?math={c}##gh-light-mode-only"> if  be the sum of the lengths of all the candidates. Then:<br>
<img src="https://render.githubusercontent.com/render/math?math={BP(Y) = 1}##gh-light-mode-only"> if <img src="https://render.githubusercontent.com/render/math?math={c > r}##gh-light-mode-only">, <img src="https://render.githubusercontent.com/render/math?math={BP(Y) = \exp\left(1 - \frac{r}{c}\right)}##gh-light-mode-only"> otherwise.

The BLEU score itself is typically a combination of modified n-gram precision for various n (usually up to 4):
<img src="https://render.githubusercontent.com/render/math?math={BLEU(Y) = BP(Y) \cdot \exp\left(\sum_{n=1}^{N} w_{n} \cdot \log\left(modified-precision(Y, n\right)\right)}##gh-light-mode-only"><br>
where <img src="https://render.githubusercontent.com/render/math?math={Y}##gh-light-mode-only"> is the dataset, and <img src="https://render.githubusercontent.com/render/math?math={w_n}##gh-light-mode-only"> is a weight for each n-gram level (usually set to 1/n).

By definition, BLEU is a corpus-level metric, since the statistics above are computed across sentences over an entire test set.
The sentence-level variant requires a smoothing strategy to counteract the effect of 0 n-gram precisions, which are more probable with shorter texts.
Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the  translation's overall quality.

It has many affinities with WER, but seeks to accommodate the fact that there are typically multiple suitable outputs for a given input.

### Inputs
- **predictions** (`list`): prediction/candidate sentences. Each prediction will be automatically tokenized into a list of tokens.
- **references** (`list`): reference sentences. Each reference will be automatically tokenized into a list of tokens.
- **max_order** (`int`): maximum n-gram order to use when computing BLEU score. Defaults to `4`.
- **smooth** (`bool`): whether or not to apply Lin et al. 2004 smoothing (i.e., smooth-BLEU/sentBLEU/ORANGE). Defaults to `False`.

### Outputs
BLEU outputs a dictionary with the following values:
- **score** (`float`): BLEU score.
- **precisions** (`list of floats`): geometric mean of n-gram precisions.
- **brevity_penalty** (`float`): brevity penalty.
- **length_ratio** (`float`): ratio of lengths.
- **translation_length** (`int`): translation length.
- **reference_length** (`int`): reference length.

### Results from popular papers
The [original BLEU paper](https://aclanthology.org/P02-1040/) (Papineni et al. 2002) compares BLEU scores of five different models on the same 500-sentence corpus. These scores ranged from 0.0527 to 0.2571.

The [Attention is All you Need paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al. 2017) got a BLEU score of 0.284 on the WMT 2014 English-to-German translation task, and 0.41 on the WMT 2014 English-to-French translation task.

## Bounds
<img src="https://render.githubusercontent.com/render/math?math={[0,1]}##gh-light-mode-only">,<br>
with 1 being the best, though with no expectation that any system will achieve 1, since even sets of human-created translations do not reach this level. Specifically, this value indicates how similar the candidate text is to  the reference texts, with values closer to 1 representing more similar texts. Few human translations will attain a  score of 1, since this would indicate that the candidate is identical to one of the reference translations. For this  reason, it is not necessary to attain a score of 1.
Scores over 30 generally reflect understandable translations. Scores over 50 generally reflect good and fluent translations.

## Examples
```python
scorer = Nlgmetricverse(metrics=load_metric("bleu"))
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
{'total_items': 2, 'empty_items': 0, 'bleu': {'score': 0.3378703280802838, 'precisions': [0.84, 0.5714285714285714, 0.35294117647058826, 0.07692307692307693], 'brevity_penalty': 1.0, 'length_ratio': 1.1818181818181819, 'translation_length': 13, 'reference_length': 11}}
```

## Limitations and bias
This metric hase multiple known limitations and biases:
- Intelligibility or grammatical correctness are not taken into account.
- BLEU compares overlap in tokens from the predictions and references, instead of comparing meaning. This can lead to discrepencies between BLEU scores and human ratings.
- BLEU scores are not comparable across different datasets, nor are they comparable across different languages.
- BLEU scores can vary greatly depending on which parameters are used to generate the scores, especially when different tokenization and normalization techniques are used. It is therefore not possible to compare BLEU scores generated using different parameters, or when these parameters are unknown.
- Shorter predicted translations achieve higher scores than longer ones, simply due to how the score is calculated. A brevity penalty is introduced to attempt to counteract this.
- [Callison-Burch et al. 2006](http://www.aclweb.org/anthology/E06-1032) criticize BLEU as a machine translation metric on the grounds that it fails to correlate with human scoring of translations. They highlight its insensitivity to n-gram order and its insensitivity to n-gram types (e.g., function vs. content words) as causes of this lack of correlation. The authors find that BLEU neither correlates with human judgment on adequacy (whether the hypothesis sentence adequately captures the meaning of the reference sentence) nor on fluency (the quality of language in the hypothesis sentence).
- [Liu et al. 2016](https://www.aclweb.org/anthology/D16-1230) specifically argue against BLEU as a metric for assessing dialogue systems, based on a lack of correlation with human judgments about dialogue coherence.
- [Reiter 2018](https://aclanthology.org/J18-3002/), in his structured review of BLEU, finds a low correlation between BLEU and human judgment.
- [Sulem et al. 2018](https://aclanthology.org/D18-1081/) examine BLEU – in the context of text simplification – on grammaticality, meaning preservation and simplicity. They report a very low, and, in some cases, negative correlation with human judgment.
- BLEU score can be misleading since several permutations of the n-grams of a sentence would get the same score as the original sentence, even though not all of the permutations would be correct or sensible. In other words, BLEU admits many spurious variants. It also penalizes correct translations if they substantially differ from the vocabulary of the references. [Zhang et al. 2004](http://www.lrec-conf.org/proceedings/lrec2004/summaries/755.htm).
  
## Citation
```bibtex
@inproceedings{PapineniRWZ02,
  author    = {Kishore Papineni and
               Salim Roukos and
               Todd Ward and
               Wei{-}Jing Zhu},
  title     = {Bleu: a Method for Automatic Evaluation of Machine Translation},
  booktitle = {Proceedings of the 40th Annual Meeting of the Association for Computational
               Linguistics, July 6-12, 2002, Philadelphia, PA, {USA}},
  pages     = {311--318},
  publisher = {{ACL}},
  year      = {2002},
  url       = {https://aclanthology.org/P02-1040/},
  doi       = {10.3115/1073083.1073135}
}
@inproceedings{lin-och-2004-orange,
    title = "{ORANGE}: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation",
    author = "Lin, Chin-Yew  and
      Och, Franz Josef",
    booktitle = "{COLING} 2004: Proceedings of the 20th International Conference on Computational Linguistics",
    month = "aug 23{--}aug 27",
    year = "2004",
    address = "Geneva, Switzerland",
    publisher = "COLING",
    url = "https://www.aclweb.org/anthology/C04-1072",
    pages = "501--507",
}
```

## Further References
- [Tensorflow implementation](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py)
