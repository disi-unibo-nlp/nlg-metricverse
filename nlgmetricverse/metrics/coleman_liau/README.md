# Metric card for Coleman-Liau Index

## Metric description
The Coleman-Liau index is a readability test developed by Meri Coleman and T. L. Liau to assess text comprehension. Its output, like the Flesch-Kincaid Grade Level and Gunning-Fog index, approximates the grade level thought necessary to comprehend the text in the United States. 

Coleman-Liau, like the ARI, but unlike most of the other indices, is based on characters rather than syllables per word. Although opinions differ on its accuracy in comparison to the syllable/word and complex word indices, computer programs count characters more easily and accurately than syllables.

The Coleman-Liau index was created to be easily calculated mechanically from hard-copy text samples. It does not require the character content of words to be analyzed, unlike syllable-based readability indices, only their length in characters. As a result, it could be used with theoretically simple mechanical scanners that only need to recognize character, word, and sentence boundaries, eliminating the need for full optical character recognition or manual keypunching.

The Coleman–Liau index is calculated with the following formula:

$CLI = 0.0588L - 0.296S - 15.8$

where $L$ is the average number of letters per 100 words and $S$ is the average number of sentences per 100 words.

### Inputs
- **predictions** (`list`): A list of strings containing the predicted sentences.
- **references** (`list`): A list of strings containing the reference sentences.

### Outputs
- **score** (`float`): The Coleman-Liau index.

### Results from popular papers

## Bounds
The Coleman-Liau index can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0,10]}##gh-light-mode-only">.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
references = ["The cat is playing on the mat.", "Today is a wonderful day"]
scorer = NLGMetricverse(metrics=load_metric("coleman_liau"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{
    "coleman_liau": { 
        "score": 6.0925 
    }
}
```
## Limitations and bias
The Coleman-Liau index is based on the number of characters per word and per sentence, and thus does not take into account the complexity of the words used. As a result, it is not a good metric for evaluating the complexity of texts in languages such as Chinese, where a single character can represent a whole word.

## Citation
```bibtex
@article{coleman1975computer,
  title={Computer readability studies},
  author={Coleman, Meri and Liau, T L},
  journal={Journal of communication},
  volume={25},
  number={4},
  pages={124--132},
  year={1975},
  publisher={Wiley Online Library}
}
```
## Further References
- [Coleman-Liau Index Wikipedia Page](https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index)