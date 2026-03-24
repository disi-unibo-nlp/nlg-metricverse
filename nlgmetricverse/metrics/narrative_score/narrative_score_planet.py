# coding=utf-8
# Copyright 2026 CittaVerse (一念万相科技)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Narrative Score metric for evaluating Chinese autobiographical narrative quality.

Implements a six-dimensional scoring framework based on autobiographical memory
research (Levine et al., 2002; Reese et al., 2011):
1. Event Richness — internal/external detail density
2. Temporal Coherence — time marker presence and distribution
3. Causal Coherence — causal reasoning markers
4. Emotional Depth — emotion vocabulary density
5. Identity Integration — self-reference markers
6. Information Density Distribution — central vs. peripheral event ratio

Designed for evaluating AI-generated reminiscence therapy narratives, life story
books, and autobiographical memory elicitation outputs in Chinese.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple

import evaluate
import numpy as np

from nlgmetricverse.collator import Collator
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """\
@software{cittaverse_narrative_scorer,
  author    = {CittaVerse},
  title     = {Narrative Scorer: Six-Dimensional Chinese Autobiographical Narrative Quality Assessment},
  year      = {2026},
  url       = {https://github.com/CittaVerse/narrative-scorer},
  version   = {0.6.0},
  note      = {Rule-based scoring for event richness, temporal/causal coherence, emotional depth, identity integration, and information density}
}
"""

_DESCRIPTION = """\
Narrative Score evaluates the quality of Chinese autobiographical narratives across six dimensions
grounded in memory science research. It is designed for assessing AI-generated or human-produced
reminiscence therapy outputs, life story books, and autobiographical memory elicitation transcripts.

The six dimensions are:
1. **Event Richness** (事件丰富度): Density of distinct events extracted from the narrative.
2. **Temporal Coherence** (时间连贯性): Presence and distribution of temporal markers.
3. **Causal Coherence** (因果连贯性): Density of causal reasoning markers (negation-aware).
4. **Emotional Depth** (情感深度): Emotion vocabulary density (negation-aware).
5. **Identity Integration** (自我认同整合): Self-reference marker density.
6. **Information Density** (信息密度分布): Ratio of central (specific, verifiable) vs. peripheral (reflective) events.

Each dimension scores 0–100. The composite score is a weighted average (default weights emphasize
emotional depth and information density at 0.20 each, others at 0.15).

This metric operates reference-free: it scores predictions directly without requiring gold references.
When references are provided, the metric scores each text independently and returns per-dimension scores.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions: list of Chinese narrative texts to score.
    references: list of reference texts (scored independently; used for comparison).
Returns:
    'score': Weighted composite score (0-100).
    'event_richness': Event richness dimension score (0-100).
    'temporal_coherence': Temporal coherence dimension score (0-100).
    'causal_coherence': Causal coherence dimension score (0-100).
    'emotional_depth': Emotional depth dimension score (0-100).
    'identity_integration': Identity integration dimension score (0-100).
    'information_density': Information density distribution score (0-100).
    'total_events': Number of events extracted.
    'central_ratio': Ratio of central (specific) events.
    'letter_grade': Letter grade (S/A/B/C/D/F).
Examples:
    >>> from nlgmetricverse import NLGMetricverse, load_metric
    >>> scorer = NLGMetricverse(metrics=load_metric("narrative_score"))
    >>> predictions = [
    ...     ["我记得那是一个春天的下午，我和奶奶一起去公园散步。奶奶讲述了她年轻时在杭州的故事，因为那时生活很艰苦，但家人在一起就很幸福。我觉得很感动。"]
    ... ]
    >>> references = [
    ...     ["那天下午阳光明媚，我和外婆去了西湖边。她回忆起1960年在工厂工作的日子，虽然辛苦但很充实。"]
    ... ]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    {'narrative_score': {'score': ..., 'event_richness': ..., ...}}
"""

# ============================================================================
# Linguistic Resources (Chinese)
# ============================================================================

TIME_MARKERS = [
    "然后", "接着", "随后", "之后", "之前", "当时", "那时", "现在",
    "今天", "昨天", "明天", "去年", "今年", "明年", "小时候", "长大后",
    "首先", "其次", "最后", "一开始", "后来", "终于", "突然", "渐渐"
]

CAUSAL_MARKERS = [
    "因为", "所以", "因此", "于是", "结果", "导致", "由于", "既然",
    "为了", "以便", "从而", "致使", "怪不得", "难怪"
]

SELF_MARKERS = ["我", "我的", "我自己", "咱", "咱们", "自己"]

EMOTION_WORDS = [
    "开心", "快乐", "高兴", "幸福", "满足", "欣慰", "骄傲", "自豪",
    "温暖", "感动", "感激", "喜欢", "爱", "兴奋", "激动", "期待",
    "难过", "伤心", "悲伤", "痛苦", "害怕", "恐惧", "焦虑", "紧张",
    "生气", "愤怒", "失望", "遗憾", "后悔", "孤独", "无助", "疲惫",
    "惊讶", "意外", "困惑", "迷茫", "平静", "放松", "复杂", "矛盾"
]

NEGATION_PREFIXES = [
    "不", "没有", "没", "未", "别", "莫", "勿",
    "不是", "不会", "不能", "不可", "无法", "并不", "并非",
    "从不", "从未", "毫不", "毫无", "绝不", "绝非"
]

NEGATION_EXCEPTIONS = ["非常", "非凡", "非但", "非得"]

NEGATION_WINDOW = 4

TOPIC_TRANSITION_MARKERS = [
    "后来", "之后", "接下来", "从那以后", "过了一段时间",
    "另外", "说到这个", "说起来", "再说", "换个话题",
    "还有一件事", "除此之外", "不过",
    "到了那里", "回到家", "到了学校",
    "最重要的是", "印象最深的是", "最难忘的是",
    "但是", "然而", "可是", "不过",
]

PLACE_MARKERS = [
    "家", "学校", "医院", "公园", "商店", "超市", "车站", "机场",
    "饭店", "餐厅", "办公室", "图书馆", "教室", "操场", "广场",
    "村", "镇", "县", "市", "省", "北京", "上海", "杭州",
    "河边", "山上", "海边", "田里", "路上"
]

PEOPLE_MARKERS = [
    "爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆",
    "哥哥", "姐姐", "弟弟", "妹妹", "老师", "同学", "朋友", "邻居",
    "丈夫", "妻子", "孩子", "儿子", "女儿", "老伴"
]

MIN_EVENT_CLAUSE_LENGTH = 5
MERGE_THRESHOLD_LENGTH = 8

DEFAULT_WEIGHTS = {
    "event_richness": 0.15,
    "temporal_coherence": 0.15,
    "causal_coherence": 0.15,
    "emotional_depth": 0.20,
    "identity_integration": 0.15,
    "information_density": 0.20,
}


# ============================================================================
# Internal Scoring Functions
# ============================================================================

def _is_negated(text: str, word_pos: int) -> bool:
    """Check if a word at word_pos is preceded by a negation prefix."""
    search_start = max(0, word_pos - NEGATION_WINDOW - max(len(p) for p in NEGATION_PREFIXES))
    prefix_text = text[search_start:word_pos]
    for neg in sorted(NEGATION_PREFIXES, key=len, reverse=True):
        neg_pos = prefix_text.rfind(neg)
        if neg_pos >= 0:
            gap = len(prefix_text) - neg_pos - len(neg)
            if gap <= NEGATION_WINDOW:
                abs_neg_pos = search_start + neg_pos
                if not any(text[abs_neg_pos:abs_neg_pos + len(exc)] == exc for exc in NEGATION_EXCEPTIONS):
                    return True
    return False


def _count_with_negation(text: str, words: list) -> Tuple[int, int]:
    """Count word occurrences, distinguishing negated from non-negated. Returns (positive, negated)."""
    positive, negated = 0, 0
    for word in words:
        start = 0
        while True:
            pos = text.find(word, start)
            if pos == -1:
                break
            if _is_negated(text, pos):
                negated += 1
            else:
                positive += 1
            start = pos + len(word)
    return positive, negated


def _extract_events(text: str) -> List[Dict]:
    """
    Event extraction v0.6 — topic-transition-aware boundary detection.
    Returns list of dicts with keys: description, event_type, time_marker, people.
    """
    if not text or not text.strip():
        return []
    text = text.strip()

    # Step 1: Split by sentence-ending punctuation
    sentences = re.split(r'[。！？!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Step 2: Sub-split by topic transition markers
    raw_clauses = []
    for sentence in sentences:
        split_positions = []
        for marker in TOPIC_TRANSITION_MARKERS:
            pos = 0
            while True:
                idx = sentence.find(marker, pos)
                if idx == -1:
                    break
                if idx == 0 or sentence[idx - 1] in '，,、 ' or idx <= 3:
                    split_positions.append(idx)
                pos = idx + len(marker)
        if split_positions:
            split_positions = sorted(set(split_positions))
            prev = 0
            for sp in split_positions:
                chunk = sentence[prev:sp].strip().rstrip('，,、 ')
                if chunk:
                    raw_clauses.append(chunk)
                prev = sp
            remainder = sentence[prev:].strip()
            if remainder:
                raw_clauses.append(remainder)
        else:
            if len(sentence) > 40 and '，' in sentence:
                comma_parts = sentence.split('，')
                if all(len(p.strip()) >= 8 for p in comma_parts if p.strip()):
                    for part in comma_parts:
                        part = part.strip()
                        if part:
                            raw_clauses.append(part)
                else:
                    raw_clauses.append(sentence)
            else:
                raw_clauses.append(sentence)

    # Step 3: Merge consecutive short clauses
    merged_clauses = []
    buffer = ""
    for clause in raw_clauses:
        if len(clause) < MIN_EVENT_CLAUSE_LENGTH and buffer:
            buffer = buffer + "，" + clause
        elif len(clause) < MIN_EVENT_CLAUSE_LENGTH and not buffer:
            buffer = clause
        else:
            if buffer:
                if len(buffer) < MERGE_THRESHOLD_LENGTH:
                    buffer = buffer + "，" + clause
                    merged_clauses.append(buffer)
                    buffer = ""
                else:
                    merged_clauses.append(buffer)
                    buffer = ""
                    merged_clauses.append(clause)
            else:
                merged_clauses.append(clause)
    if buffer:
        if merged_clauses and len(buffer) < MERGE_THRESHOLD_LENGTH:
            merged_clauses[-1] = merged_clauses[-1] + "，" + buffer
        else:
            merged_clauses.append(buffer)

    # Step 4: Build events with enhanced classification
    events = []
    for clause in merged_clauses:
        if len(clause) < 3:
            continue
        if not re.search(r'[\u4e00-\u9fff\w]', clause):
            continue

        # Classify
        specifics_score = 0
        if re.search(r'\d+|[一二三四五六七八九十百千万]', clause):
            specifics_score += 2
        if any(place in clause for place in PLACE_MARKERS):
            specifics_score += 2
        if any(person in clause for person in PEOPLE_MARKERS):
            specifics_score += 1
        action_verbs = ["去了", "来了", "做了", "买了", "吃了", "看了", "说了",
                        "走了", "跑了", "拿了", "给了", "送了", "搬了", "带了"]
        if any(verb in clause for verb in action_verbs):
            specifics_score += 1
        reflection_markers = ["觉得", "认为", "想", "感觉", "可能", "也许",
                              "大概", "好像", "似乎", "总之", "总的来说",
                              "回想起来", "现在看来", "我想"]
        if any(word in clause for word in reflection_markers):
            specifics_score -= 2

        event_type = "central" if specifics_score >= 1 else "peripheral"
        time_marker = next((m for m in TIME_MARKERS if m in clause), None)
        people = [p for p in PEOPLE_MARKERS if p in clause]

        events.append({
            "description": clause,
            "event_type": event_type,
            "time_marker": time_marker,
            "people": people,
        })
    return events


def _score_text(text: str) -> Dict:
    """Score a single Chinese narrative text. Returns dict of all dimension scores."""
    text = text.strip()
    text_length = len(text)

    events = _extract_events(text)
    time_markers_count = sum(text.count(m) for m in TIME_MARKERS)

    # Causal markers (negation-aware: negated discounted 50%)
    pos_causal, neg_causal = _count_with_negation(text, CAUSAL_MARKERS)
    causal_markers_count = pos_causal + (neg_causal // 2)

    self_references_count = sum(text.count(m) for m in SELF_MARKERS)

    # Emotion words (negation-aware: negated emotions still count)
    pos_emo, neg_emo = _count_with_negation(text, EMOTION_WORDS)
    emotion_words_count = pos_emo + neg_emo

    central_count = sum(1 for e in events if e["event_type"] == "central")
    peripheral_count = len(events) - central_count
    central_ratio = central_count / len(events) if events else 0.0

    # Dimension 1: Event Richness
    if events and text_length > 0:
        events_per_100 = (len(events) / text_length) * 100
        event_richness = min(events_per_100 * 10, 100.0)
    else:
        event_richness = 0.0

    # Dimension 2: Temporal Coherence
    if events and text_length > 0:
        marker_density = (time_markers_count / text_length) * 100
        events_with_time = sum(1 for e in events if e["time_marker"])
        time_coverage = (events_with_time / len(events)) * 100
        temporal_coherence = min(marker_density * 20, 50) + time_coverage * 0.5
        temporal_coherence = min(temporal_coherence, 100.0)
    else:
        temporal_coherence = 0.0

    # Dimension 3: Causal Coherence
    if text_length > 0:
        causal_density = (causal_markers_count / text_length) * 100
        causal_coherence = min(causal_density * 20, 100.0)
    else:
        causal_coherence = 0.0

    # Dimension 4: Emotional Depth
    if text_length > 0:
        emotion_density = (emotion_words_count / text_length) * 100
        emotional_depth = min(emotion_density * 33, 100.0)
    else:
        emotional_depth = 0.0

    # Dimension 5: Identity Integration
    if text_length > 0:
        self_density = (self_references_count / text_length) * 100
        identity_integration = min(self_density * 50, 100.0)
    else:
        identity_integration = 0.0

    # Dimension 6: Information Density Distribution
    if events:
        optimal_ratio = 0.6
        deviation = abs(central_ratio - optimal_ratio)
        information_density = max(0, 100 - (deviation * 200))
    else:
        information_density = 0.0

    # Composite
    scores = {
        "event_richness": round(event_richness, 2),
        "temporal_coherence": round(temporal_coherence, 2),
        "causal_coherence": round(causal_coherence, 2),
        "emotional_depth": round(emotional_depth, 2),
        "identity_integration": round(identity_integration, 2),
        "information_density": round(information_density, 2),
    }
    composite = sum(scores[dim] * DEFAULT_WEIGHTS[dim] for dim in scores)
    composite = round(composite, 2)

    # Letter grade
    if composite >= 90:
        grade = "S"
    elif composite >= 80:
        grade = "A"
    elif composite >= 70:
        grade = "B"
    elif composite >= 60:
        grade = "C"
    elif composite >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": composite,
        "event_richness": scores["event_richness"],
        "temporal_coherence": scores["temporal_coherence"],
        "causal_coherence": scores["causal_coherence"],
        "emotional_depth": scores["emotional_depth"],
        "identity_integration": scores["identity_integration"],
        "information_density": scores["information_density"],
        "total_events": len(events),
        "central_count": central_count,
        "peripheral_count": peripheral_count,
        "central_ratio": round(central_ratio, 3),
        "letter_grade": grade,
    }


# ============================================================================
# NLG Metricverse Plugin
# ============================================================================

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NarrativeScorePlanet(MetricForLanguageGeneration):
    """
    Narrative Score metric for nlg-metricverse.

    Evaluates Chinese autobiographical narrative quality across six dimensions.
    This is a reference-free metric: predictions are scored independently based on
    linguistic features. When references are provided, both are scored and the
    prediction score is returned as the primary result.
    """

    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=100,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/CittaVerse/narrative-scorer"],
            reference_urls=[
                "https://doi.org/10.1016/S0028-3932(02)00017-7",  # Levine et al., 2002
                "https://doi.org/10.1080/09658211.2011.578995",   # Reese et al., 2011
            ],
        )

    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        """Score each prediction text; references are scored for comparison but prediction score is primary."""
        pred_scores = []
        for pred in predictions:
            result = _score_text(pred)
            pred_scores.append(result["score"])

        avg_score = sum(pred_scores) / len(pred_scores) if pred_scores else 0.0

        # Also provide full dimensional breakdown for the corpus
        all_results = [_score_text(pred) for pred in predictions]
        n = len(all_results)
        if n == 0:
            return {"score": 0.0}

        aggregated = {
            "score": round(avg_score, 2),
            "event_richness": round(sum(r["event_richness"] for r in all_results) / n, 2),
            "temporal_coherence": round(sum(r["temporal_coherence"] for r in all_results) / n, 2),
            "causal_coherence": round(sum(r["causal_coherence"] for r in all_results) / n, 2),
            "emotional_depth": round(sum(r["emotional_depth"] for r in all_results) / n, 2),
            "identity_integration": round(sum(r["identity_integration"] for r in all_results) / n, 2),
            "information_density": round(sum(r["information_density"] for r in all_results) / n, 2),
            "total_events": round(sum(r["total_events"] for r in all_results) / n, 2),
            "central_ratio": round(sum(r["central_ratio"] for r in all_results) / n, 3),
        }
        return aggregated

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        """
        Single prediction, multiple references.
        Since this is reference-free, we score predictions directly.
        """
        return self._compute_single_pred_single_ref(
            predictions=predictions,
            references=Collator([references[0][0]] if references else [], keep=True),
            reduce_fn=reduce_fn,
            **kwargs,
        )

    def _compute_multi_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        """
        Multiple predictions, multiple references.
        Score each prediction independently, aggregate.
        """
        scores = []
        for preds, refs in zip(predictions, references):
            pred_score = self._compute_single_pred_single_ref(
                Collator([preds[0]] if isinstance(preds, list) else [preds], keep=True),
                Collator([refs[0]] if isinstance(refs, list) else [refs], keep=True),
                reduce_fn=reduce_fn,
            )
            scores.append(pred_score)

        if not scores:
            return {"score": 0.0}

        return self._reduce_scores(scores, reduce_fn=reduce_fn or np.mean)
