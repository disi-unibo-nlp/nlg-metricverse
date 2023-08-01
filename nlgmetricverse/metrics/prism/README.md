# Metric Card for Prism


## Metric Description
Prism is an automatic MT metric which uses a sequence-to-sequence paraphraser to score MT system outputs conditioned on their respective
human references.  Prism uses a multilingual NMT model as a zero-shot paraphraser, which negates the need for synthetic paraphrase data and
results in a single model which works in many languages.
Prism outperforms or statistically ties with all metrics submitted to the [WMT 2019 metrics shared task](https://www.aclweb.org/anthology/W19-5302/) as segment-level human correlation.

The official library provides a large, pre-trained multilingual NMT model which is used as a multilingual paraphraser, but the model may also be of use to the research community beyond MT metrics.

Prism scores raw, untokenized text; all preprocessing is applied internally.

# Metric Usage: Python Module

All functionality is also available in Python, for example:

```python
import os
from prism import Prism
prism = Prism(model_dir=os.environ['MODEL_DIR'], lang='en')
print('Prism identifier:', prism.identifier())
cand = ['Hi world.', 'This is a Test.']
ref = ['Hello world.', 'This is a test.']
src = ['Bonjour le monde.', "C'est un test."]
print('System-level metric:', prism.score(cand=cand, ref=ref))
print('Segment-level metric:', prism.score(cand=cand, ref=ref, segment_scores=True))
print('System-level QE-as-metric:', prism.score(cand=cand, src=src))
print('Segment-level QE-as-metric:', prism.score(cand=cand, src=src, segment_scores=True))
```

Which should produce:

>Prism identifier: {'version': '0.1', 'model': 'm39v1', 'seg_scores': 'avg_log_prob', 'sys_scores': 'avg_log_prob', 'log_base': 2}
>System-level metric: -1.0184666  
>Segment-level metric: [-1.4878583 -0.5490748]  
>System-level QE-as-metric: -1.8306842  
>Segment-level QE-as-metric: [-2.462842  -1.1985264]  

## Multilingual Translation
The Prism model is simply a multilingual NMT model, and can be used for translation --  see the [multilingual translation README](translation/README.md).

## Paraphrase Generation

Attempting to generate paraphrases from the Prism model via naive beam search
(e.g. "translate" from French to French) results in trivial copies most of the time.
However, we provide a simple algorithm to discourage copying
and enable paraphrase generation in many languages -- see the [paraphrase generation README](paraphrase_generation/README.md).


## Supported Languages

Albanian (sq), Arabic (ar), Bengali (bn), Bulgarian (bg), 
Catalan; Valencian (ca), Chinese (zh), Croatian (hr), Czech (cs), 
Danish (da), Dutch (nl), English (en), Esperanto (eo), Estonian (et),
Finnish (fi),  French (fr), German (de), Greek, Modern (el),
Hebrew (modern) (he),  Hungarian (hu), Indonesian (id), Italian (it),
Japanese (ja), Kazakh (kk), Latvian (lv), Lithuanian (lt), Macedonian (mk),
Norwegian (no), Polish (pl), Portuguese (pt), Romanian, Moldavan (ro),
Russian (ru), Serbian (sr), Slovak (sk), Slovene (sl), Spanish; Castilian (es),
Swedish (sv), Turkish (tr), Ukrainian (uk), Vietnamese (vi)

## Data Filtering

The data filtering scripts used to train the Prism model can be found [here](https://github.com/thompsonb/prism_bitext_filter).

## Citation
```bibtex
@inproceedings{thompson-post-2020-automatic,
    title={Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing},
    author={Brian Thompson and Matt Post},
    year={2020},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```

## Further References
- [WMT paper](https://aclanthology.org/2020.wmt-1.67/)
