import nlgmetricverse, os

comet_metric = nlgmetricverse.load_metric('comet', config_name="wmt21-cometinho-da")
predictions = os.getcwd() + "/predictions"
references = os.getcwd() + "/references"
sources = os.getcwd() + "/sources"

results = comet_metric.compute(sources=sources, predictions=predictions, references=references)

print(results)
