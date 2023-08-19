# Metric card for UNR

## Metric description
The UNR metric measures the summary of n-grams uniqueness: <br><img src="https://render.githubusercontent.com/render/math?math={\frac{count(uniq_n_gram(y))}{count(n_gram(y))}}"><br>
  
Where we take <img src="https://render.githubusercontent.com/render/math?math={n \in [1,3]}"> and divide the average by variance.

### Inputs
- **predictions** (istance of EvaluationInstance): An object containing the predicted text.
- **references** (istance of EvaluationInstance): An object containing the reference text.

### Outputs
- **UNR**(`float` or `int`): UNR score. Minimum possible value is 0. Maximum possible value is 1, the higher the score it means the more unique the summary is.

### Results from popular papers

## Bounds
The `UNR` score has a <img src="https://render.githubusercontent.com/render/math?math={[0,1]}"> range.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
references = ["The cat is playing on the mat.", "Today is a wonderful day"]
scorer = NLGMetricverse(metrics=load_metric("unr"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "unr": { 
    'unr_1': 0.9,
    'unr_2': 0.9444,
    'unr_3': 1.0,
    'unr_avg': 0.9481333333333334,
  } 
}
```

## Limitations and Bias

## Citation

## Further References