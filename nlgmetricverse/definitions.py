DEFAULT_METRICS = [
    {"path": "bertscore", "compute_kwargs": {"idf": True, "rescale_with_baseline": True, "model_type": "microsoft/deberta-xlarge-mnli"}},
    {"path": "bartscore"},
    {"path": "meteor"},
    {"path": "rouge"}
]
