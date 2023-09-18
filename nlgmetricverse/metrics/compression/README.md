# Metric card for Compression

## Metric description
The Compression metric calculates the documention-summary word ration: $$\frac{|x|}{|y|}$$

Summarizing with higher compression is challenging as it requires capturing more precisely the critical aspects of the article text. It has been observed that publications with lower compression ratios exhibit higher diversity along both dimensions of extractiveness. However, as the median compression ratio increases, the distributions become more concentrated, indicating that summarization strategies become more rigid.

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.

### Outputs
- **compression**(`float` or `int`): Compression score. Minimum possible value is 0. Maximum possible value is |x|, where the higher the score, means the more compression the summary is.

### Results from popular papers

## Bounds
The `compression` score has a $[0,|x|],\uparrow$ range.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
scorer = NLGMetricverse(metrics=load_metric("compression"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "compression": { 
    'score' : 0.95 
  } 
}
```

## Limitations and Bias

## Citation
```bibtex
@article{grusky2018newsroom,
  title={Newsroom: A dataset of 1.3 million summaries with diverse extractive strategies},
  author={Grusky, Max and Naaman, Mor and Artzi, Yoav},
  journal={arXiv preprint arXiv:1804.11283},
  year={2018}
}
```

## Further References

## Contributions
Thanks to @ValentinaPieri for contributing to this metric!