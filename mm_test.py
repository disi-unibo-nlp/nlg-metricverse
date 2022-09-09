import nlgmetricverse
scorer = nlgmetricverse.load_metric("character")
ref=[["hi there is a good day", "second good day of the week"], ["there is a problem", "there are a problems"]]
hyp=[["yes there is a bad day", "bad bad"], ["hi, I'm a person", "good, day"]]

scorer.compute(references=ref, predictions=hyp, reduce_fn="max")
