# coding=utf-8
"""
Tests for the Narrative Score metric plugin.
"""
import unittest
import sys
import os

# Add the project root to path so we can import the metric directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nlgmetricverse.metrics.narrative_score.narrative_score_planet import (
    _score_text,
    _extract_events,
    _count_with_negation,
    _is_negated,
    EMOTION_WORDS,
    CAUSAL_MARKERS,
    NarrativeScorePlanet,
)


class TestScoreText(unittest.TestCase):
    """Test the core _score_text function."""

    def test_basic_narrative(self):
        text = "我记得那是一个春天的下午，我和奶奶一起去公园散步。奶奶开始讲述她年轻时的故事。我觉得很感动。"
        result = _score_text(text)
        self.assertIn("score", result)
        self.assertGreater(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertIn("letter_grade", result)
        self.assertIn(result["letter_grade"], ["S", "A", "B", "C", "D", "F"])

    def test_empty_text(self):
        result = _score_text("")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["total_events"], 0)

    def test_rich_narrative(self):
        text = (
            "1985年的夏天，我和爸爸去了北京。那时我只有10岁，"
            "因为第一次坐火车所以非常兴奋。到了那里，我们先去了天安门广场，"
            "后来又去了故宫。爸爸给我买了一个冰棍，我开心极了。"
            "回到家后，我把照片给妈妈看，她也很高兴。"
            "我觉得那是我童年最幸福的记忆。"
        )
        result = _score_text(text)
        # Rich narrative should score higher
        self.assertGreater(result["score"], 30)
        self.assertGreater(result["total_events"], 2)
        self.assertGreater(result["event_richness"], 0)
        self.assertGreater(result["emotional_depth"], 0)
        self.assertGreater(result["identity_integration"], 0)

    def test_all_dimensions_present(self):
        text = "然后我和妈妈去了学校，因为要开家长会。我觉得很紧张，但后来感觉还好。"
        result = _score_text(text)
        for dim in ["event_richness", "temporal_coherence", "causal_coherence",
                     "emotional_depth", "identity_integration", "information_density"]:
            self.assertIn(dim, result)
            self.assertGreaterEqual(result[dim], 0)
            self.assertLessEqual(result[dim], 100)

    def test_central_ratio_range(self):
        text = "我去了北京的公园。我觉得还不错。"
        result = _score_text(text)
        self.assertGreaterEqual(result["central_ratio"], 0.0)
        self.assertLessEqual(result["central_ratio"], 1.0)


class TestEventExtraction(unittest.TestCase):
    """Test event boundary detection."""

    def test_empty(self):
        self.assertEqual(len(_extract_events("")), 0)

    def test_single_sentence(self):
        events = _extract_events("我去了北京。")
        self.assertGreaterEqual(len(events), 1)

    def test_transition_marker_split(self):
        events = _extract_events("我在家里看书，后来去了公园散步。")
        self.assertGreaterEqual(len(events), 1)

    def test_multiple_sentences(self):
        text = "我先去了学校。然后去了图书馆。最后回到家。"
        events = _extract_events(text)
        self.assertGreaterEqual(len(events), 2)

    def test_classification_central(self):
        events = _extract_events("1985年爸爸带我去了北京天安门。")
        if events:
            # Has year, place, person → should be central
            self.assertEqual(events[0]["event_type"], "central")

    def test_classification_peripheral(self):
        events = _extract_events("我觉得可能那段时间过得还不错。")
        if events:
            self.assertEqual(events[0]["event_type"], "peripheral")


class TestNegation(unittest.TestCase):
    """Test negation detection."""

    def test_negated_emotion(self):
        pos, neg = _count_with_negation("我不开心", EMOTION_WORDS)
        self.assertEqual(neg, 1)
        self.assertEqual(pos, 0)

    def test_positive_emotion(self):
        pos, neg = _count_with_negation("我很开心", EMOTION_WORDS)
        self.assertEqual(pos, 1)
        self.assertEqual(neg, 0)

    def test_negation_exception(self):
        # "非常" is NOT a negation
        pos, neg = _count_with_negation("非常开心", EMOTION_WORDS)
        self.assertEqual(pos, 1)
        self.assertEqual(neg, 0)

    def test_negated_causal(self):
        pos, neg = _count_with_negation("没有因为下雨", CAUSAL_MARKERS)
        self.assertEqual(neg, 1)


class TestLetterGrade(unittest.TestCase):
    """Test grading system."""

    def test_high_score(self):
        # A very rich narrative should get at least C
        text = (
            "1960年的冬天，我和妈妈去了上海的外婆家。因为那年特别冷，"
            "所以我们穿了很多衣服。外婆非常高兴地迎接我们，她做了很多好吃的。"
            "我记得那天晚上，我们围着炉火聊天，外婆讲起了她年轻时的故事。"
            "她说那时候在工厂工作，虽然辛苦但很充实。我觉得很感动，"
            "因为外婆平时很少说这些。后来我们一起包了饺子，我包得很丑，"
            "但是外婆说味道最重要。那是我记忆中最温暖的一个冬天。"
        )
        result = _score_text(text)
        self.assertIn(result["letter_grade"], ["S", "A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
