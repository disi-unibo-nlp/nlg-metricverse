# Metric Card for Narrative Score

## Metric Description

Narrative Score evaluates the quality of **Chinese autobiographical narratives** across six dimensions grounded in memory science research (Levine et al., 2002; Reese et al., 2011). It is designed for assessing AI-generated or human-produced reminiscence therapy outputs, life story books, and autobiographical memory elicitation transcripts.

The six dimensions are:
1. **Event Richness** (事件丰富度) — Density of distinct events extracted via topic-transition-aware boundary detection.
2. **Temporal Coherence** (时间连贯性) — Presence and distribution of 24 Chinese temporal markers.
3. **Causal Coherence** (因果连贯性) — Density of causal reasoning markers (negation-aware: negated causals discounted 50%).
4. **Emotional Depth** (情感深度) — Emotion vocabulary density across 40 positive/negative/complex emotion words (negation-aware).
5. **Identity Integration** (自我认同整合) — Self-reference marker density.
6. **Information Density Distribution** (信息密度分布) — Ratio of central (specific, verifiable) vs. peripheral (reflective) events, optimized around 60:40.

**This is a reference-free metric**: it scores predictions based on linguistic features without requiring gold-standard references.

### Inputs
- **predictions** (`list` of `str`): Chinese narrative texts to score.
- **references** (`list` of `str`): Reference texts (scored independently for comparison; not required for primary scoring).

### Outputs
- **score** (`float`): Weighted composite score (0–100).
- **event_richness** (`float`): Event richness dimension (0–100).
- **temporal_coherence** (`float`): Temporal coherence dimension (0–100).
- **causal_coherence** (`float`): Causal coherence dimension (0–100).
- **emotional_depth** (`float`): Emotional depth dimension (0–100).
- **identity_integration** (`float`): Identity integration dimension (0–100).
- **information_density** (`float`): Information density distribution (0–100).
- **total_events** (`float`): Average number of events extracted per text.
- **central_ratio** (`float`): Average ratio of central (specific) events.

### Results from Popular Papers

The scoring dimensions are aligned with:
- **Autobiographical Interview** (Levine et al., 2002): Internal/external detail scoring for autobiographical memory.
- **Narrative Coherence Coding Scheme** (Reese et al., 2011): Context, chronology, and theme coherence.
- **Autobiographical Memory Test** (Williams & Broadbent, 1986): Specificity of memory retrieval.

## Bounds
All dimension scores and the composite score range from **0 to 100**.
- **0**: No narrative features detected.
- **100**: Maximum density of narrative features.
- **Letter grades**: S (≥90), A (≥80), B (≥70), C (≥60), D (≥50), F (<50).

## Examples

```python
from nlgmetricverse import NLGMetricverse, load_metric

scorer = NLGMetricverse(metrics=load_metric("narrative_score"))

predictions = [
    ["我记得那是一个春天的下午，阳光明媚。我和奶奶一起去公园散步，那是她最喜欢的地方。我们坐在湖边的长椅上，奶奶开始讲述她年轻时的故事。她说那时候生活很艰苦，因为要照顾五个孩子，但家人在一起就很幸福。我觉得很感动，因为奶奶很少提起过去的事情。那天我们聊了很多，我感觉更了解她了。"]
]

references = [
    ["那天下午阳光明媚，我和外婆去了西湖边。她回忆起1960年在工厂工作的日子，虽然辛苦但很充实。"]
]

scores = scorer(predictions=predictions, references=references)
print(scores)
# {
#   "narrative_score": {
#     "score": 65.42,
#     "event_richness": 45.0,
#     "temporal_coherence": 52.3,
#     "causal_coherence": 38.1,
#     "emotional_depth": 72.6,
#     "identity_integration": 88.5,
#     "information_density": 80.0,
#     "total_events": 5,
#     "central_ratio": 0.6
#   }
# }
```

## Limitations and Bias

- **Language-specific**: Currently designed for **Chinese (Simplified)** narratives only. Marker lists would need adaptation for other languages.
- **Rule-based**: Does not use neural models; relies on keyword matching and heuristics. More nuanced narrative qualities (irony, metaphor, narrative arc) are not captured.
- **Negation handling**: Simple prefix-based negation detection with a 4-character window; complex negation structures may be missed.
- **Event boundary detection**: Topic-transition-aware but not semantic; ambiguous clause boundaries may lead to over/under-segmentation.
- **Cultural specificity**: Emotion word lists and place/people markers reflect mainland Chinese cultural context.

## Citations

```bibtex
@software{cittaverse_narrative_scorer,
  author    = {CittaVerse},
  title     = {Narrative Scorer: Six-Dimensional Chinese Autobiographical Narrative Quality Assessment},
  year      = {2026},
  url       = {https://github.com/CittaVerse/narrative-scorer},
  version   = {0.6.0}
}

@article{levine2002aging,
  title     = {Aging and autobiographical memory: Dissociating episodic from semantic retrieval},
  author    = {Levine, Brian and Svoboda, Eva and Hay, Janine F and Winocur, Gordon and Moscovitch, Morris},
  journal   = {Psychology and aging},
  volume    = {17},
  number    = {4},
  pages     = {677},
  year      = {2002},
  publisher = {American Psychological Association}
}

@article{reese2011coherence,
  title     = {Coherence of personal narratives across the lifespan: A multidimensional model and coding method},
  author    = {Reese, Elaine and Haden, Catherine A and Baker-Ward, Lynne and Bauer, Patricia and Fivush, Robyn and Ornstein, Peter A},
  journal   = {Journal of cognition and development},
  volume    = {12},
  number    = {4},
  pages     = {424--462},
  year      = {2011},
  publisher = {Taylor \& Francis}
}
```

## Further References
- [CittaVerse Narrative Scorer](https://github.com/CittaVerse/narrative-scorer) — Full standalone scorer with tests and CI.
- [Autobiographical Interview Scoring Manual](https://doi.org/10.1016/S0028-3932(02)00017-7)

## Contributions
Thanks to @CittaVerse for contributing this metric!
