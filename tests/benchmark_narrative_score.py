# Narrative Scoring Benchmark v1.0
# Gold-standard hand-annotated Chinese autobiographical narrative samples
# For evaluating CittaVerse Narrative Scorer v0.6.0 accuracy
#
# Each sample has:
# - text: the narrative
# - gold_events: hand-annotated event boundaries (list of event descriptions)
# - gold_event_types: central/peripheral for each event
# - gold_dimensions: expected approximate dimension scores (±10 tolerance)
# - notes: annotation rationale

import json

BENCHMARK_SAMPLES = [
    {
        "id": "bench-001",
        "label": "Rich childhood memory with specifics",
        "text": (
            "1985年的夏天，我和爸爸坐了12个小时的火车去北京。"
            "那时我只有10岁，第一次离开杭州。"
            "到了北京站，爸爸带我坐了公共汽车去天安门广场。"
            "我记得广场上人很多，有人在放风筝。"
            "后来我们去了故宫，爸爸给我讲了很多历史故事。"
            "因为走了太多路，我的脚起了水泡，但我还是很开心。"
            "晚上回到旅馆，爸爸给我买了北京烤鸭，那是我吃过最好吃的东西。"
            "我觉得那次旅行改变了我对世界的看法。"
        ),
        "gold_events": [
            {"desc": "1985年坐火车去北京", "type": "central"},
            {"desc": "10岁第一次离开杭州", "type": "central"},
            {"desc": "到北京站坐公共汽车去天安门", "type": "central"},
            {"desc": "广场上人多有人放风筝", "type": "central"},
            {"desc": "去故宫爸爸讲历史故事", "type": "central"},
            {"desc": "走路起水泡但开心", "type": "central"},
            {"desc": "旅馆吃北京烤鸭", "type": "central"},
            {"desc": "旅行改变看法（反思）", "type": "peripheral"},
        ],
        "gold_dimensions": {
            "event_richness": {"min": 40, "max": 80},
            "temporal_coherence": {"min": 30, "max": 70},
            "causal_coherence": {"min": 10, "max": 40},
            "emotional_depth": {"min": 20, "max": 60},
            "identity_integration": {"min": 40, "max": 90},
            "information_density": {"min": 50, "max": 100},
        },
        "notes": "7 central + 1 peripheral. Rich in specifics (dates, numbers, places, people). Strong identity references."
    },
    {
        "id": "bench-002",
        "label": "Sparse reflective narrative (low specificity)",
        "text": (
            "我觉得小时候的日子还不错。"
            "大概是因为那时候没什么烦恼吧。"
            "我想可能每个人都会怀念童年。"
            "总的来说那段时间过得挺快的。"
        ),
        "gold_events": [
            {"desc": "小时候还不错", "type": "peripheral"},
            {"desc": "没什么烦恼", "type": "peripheral"},
            {"desc": "每个人怀念童年", "type": "peripheral"},
            {"desc": "时间过得快", "type": "peripheral"},
        ],
        "gold_dimensions": {
            "event_richness": {"min": 10, "max": 50},
            "temporal_coherence": {"min": 0, "max": 30},
            "causal_coherence": {"min": 5, "max": 35},
            "emotional_depth": {"min": 0, "max": 20},
            "identity_integration": {"min": 30, "max": 80},
            "information_density": {"min": 0, "max": 40},
        },
        "notes": "All peripheral. No specifics, no places, no people names. Heavy reflection markers."
    },
    {
        "id": "bench-003",
        "label": "Emotional family narrative with causal reasoning",
        "text": (
            "奶奶去世那天是2003年的秋天。"
            "因为她生病很久了，所以我们全家都有心理准备。"
            "但是真正到了那一天，我还是非常难过。"
            "妈妈一直在哭，爸爸沉默不语。"
            "我记得奶奶最后对我说的话是让我好好读书。"
            "从那以后我每年清明都会去看她。"
            "我觉得奶奶教会了我什么是坚强。"
        ),
        "gold_events": [
            {"desc": "2003年奶奶去世", "type": "central"},
            {"desc": "生病很久全家有准备", "type": "central"},
            {"desc": "那天还是很难过", "type": "central"},
            {"desc": "妈妈哭爸爸沉默", "type": "central"},
            {"desc": "奶奶最后的话", "type": "central"},
            {"desc": "每年清明去看她", "type": "central"},
            {"desc": "奶奶教会坚强（反思）", "type": "peripheral"},
        ],
        "gold_dimensions": {
            "event_richness": {"min": 30, "max": 70},
            "temporal_coherence": {"min": 20, "max": 60},
            "causal_coherence": {"min": 20, "max": 60},
            "emotional_depth": {"min": 40, "max": 90},
            "identity_integration": {"min": 50, "max": 100},
            "information_density": {"min": 50, "max": 100},
        },
        "notes": "Strong emotional content (难过, 哭). Good causal chains (因为...所以). Multiple family members. Date reference."
    },
    {
        "id": "bench-004",
        "label": "Single-sentence minimal narrative",
        "text": "我小时候在村里长大。",
        "gold_events": [
            {"desc": "小时候在村里长大", "type": "central"},
        ],
        "gold_dimensions": {
            "event_richness": {"min": 30, "max": 100},
            "temporal_coherence": {"min": 10, "max": 70},
            "causal_coherence": {"min": 0, "max": 10},
            "emotional_depth": {"min": 0, "max": 10},
            "identity_integration": {"min": 20, "max": 100},
            "information_density": {"min": 0, "max": 60},
        },
        "notes": "Minimal. One event with place marker. No emotions, no causal markers."
    },
    {
        "id": "bench-005",
        "label": "Multi-scene journey with emotional arc",
        "text": (
            "去年春天，我带着孩子回到了我长大的老家。"
            "一开始我很兴奋，因为已经离开了20年。"
            "到了那里，我发现老房子已经拆了，变成了一片空地。"
            "我感到非常失望和伤心。"
            "后来我们去了河边，那条小河还在，水还是那么清。"
            "孩子在河边玩得很开心，捡了很多石头。"
            "回到家后我一个人坐了很久，想着这些年的变化。"
            "虽然有些遗憾，但我觉得带孩子来看看还是值得的。"
            "因为我希望他能知道我从哪里来。"
        ),
        "gold_events": [
            {"desc": "去年带孩子回老家", "type": "central"},
            {"desc": "兴奋因为离开20年", "type": "central"},
            {"desc": "老房子拆了变空地", "type": "central"},
            {"desc": "失望和伤心", "type": "peripheral"},
            {"desc": "去河边小河还在", "type": "central"},
            {"desc": "孩子河边玩捡石头", "type": "central"},
            {"desc": "回家一个人坐很久", "type": "peripheral"},
            {"desc": "有遗憾但值得", "type": "peripheral"},
            {"desc": "希望孩子知道从哪里来", "type": "peripheral"},
        ],
        "gold_dimensions": {
            "event_richness": {"min": 30, "max": 65},
            "temporal_coherence": {"min": 25, "max": 65},
            "causal_coherence": {"min": 15, "max": 50},
            "emotional_depth": {"min": 40, "max": 90},
            "identity_integration": {"min": 50, "max": 100},
            "information_density": {"min": 40, "max": 90},
        },
        "notes": "Rich emotional arc (兴奋→失望→遗憾→值得). Multiple scenes. Causal reasoning. Temporal progression. Mix of central/peripheral."
    },
]


def run_benchmark():
    """Run the benchmark and report results."""
    import sys
    import os
    import re
    from typing import Dict, List, Optional, Tuple

    # Load scoring functions
    planet_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'nlgmetricverse', 'metrics',
        'narrative_score', 'narrative_score_planet.py'
    )
    with open(planet_path) as f:
        source = f.read()

    marker = "# ============================================================================\n# NLG Metricverse Plugin"
    code = source.split(marker)[0]
    code = code.replace("import evaluate", "# skip")
    code = code.replace("import numpy as np", "# skip")
    code = code.replace("from nlgmetricverse.collator import Collator", "# skip")
    code = code.replace("from nlgmetricverse.utils.metric_info import MetricInfo", "# skip")
    code = code.replace("from nlgmetricverse.metrics._core import MetricForLanguageGeneration", "# skip")

    ns = {"re": re, "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple}
    exec(code, ns)
    _score_text = ns["_score_text"]
    _extract_events = ns["_extract_events"]

    print("=" * 70)
    print("CittaVerse Narrative Scorer v0.6.0 — Benchmark Results")
    print("=" * 70)

    total_dim_tests = 0
    passed_dim_tests = 0
    total_event_accuracy = 0
    sample_count = 0

    for sample in BENCHMARK_SAMPLES:
        print(f"\n{'─' * 60}")
        print(f"[{sample['id']}] {sample['label']}")
        print(f"{'─' * 60}")

        result = _score_text(sample["text"])
        events = _extract_events(sample["text"])

        # Event count comparison
        gold_count = len(sample["gold_events"])
        extracted_count = result["total_events"]
        event_diff = abs(gold_count - extracted_count)
        event_accuracy = max(0, 1 - event_diff / max(gold_count, 1))
        total_event_accuracy += event_accuracy
        sample_count += 1

        print(f"  Events: extracted={extracted_count}, gold={gold_count}, accuracy={event_accuracy:.1%}")
        print(f"  Score: {result['score']} ({result['letter_grade']})")

        # Dimension score checking
        for dim, bounds in sample["gold_dimensions"].items():
            actual = result[dim]
            in_range = bounds["min"] <= actual <= bounds["max"]
            total_dim_tests += 1
            if in_range:
                passed_dim_tests += 1
            status = "✅" if in_range else "❌"
            print(f"  {status} {dim}: {actual:.1f} (expected {bounds['min']}-{bounds['max']})")

        # Event type distribution
        gold_central = sum(1 for e in sample["gold_events"] if e["type"] == "central")
        gold_peripheral = gold_count - gold_central
        print(f"  Central/Peripheral: extracted={result['central_count']}/{result['peripheral_count']}, "
              f"gold={gold_central}/{gold_peripheral}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Dimension score accuracy: {passed_dim_tests}/{total_dim_tests} ({passed_dim_tests/total_dim_tests:.1%})")
    print(f"  Event count accuracy: {total_event_accuracy/sample_count:.1%}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_benchmark()
