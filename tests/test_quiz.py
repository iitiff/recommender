from __future__ import annotations

import pytest

from recsys.core.types import Tag
from recsys.profile import QuizSpec, build_profile
from tests.conftest import DEMO_DIR


@pytest.fixture(scope="module")
def spec():
    return QuizSpec.from_yaml(DEMO_DIR / "quiz.yaml")


def test_demo_quiz_matches_demo_catalog(spec, demo_catalog):
    assert spec.problems(demo_catalog.items) == []


def test_answers_become_likes_affinities_and_budget(spec):
    profile = build_profile(
        spec,
        {
            "known_wines": {"demo-granite-bay-30": 1, "demo-ironwood-32": -1, "demo-kestrel-34": 0},
            "style": "bold",
            "budget": "from_30_to_70",
        },
    )
    assert profile.liked == ("demo-granite-bay-30",)
    assert profile.disliked == ("demo-ironwood-32",)
    assert profile.tag_affinities[Tag("body", "full")] == 1.0
    assert (profile.min_price, profile.max_price) == (30, 70)
    assert profile.quiz_version == "demo-v1"


def test_partial_quiz_is_fine(spec):
    profile = build_profile(spec, {"style": "bright"})
    assert profile.liked == () and profile.max_price is None
    assert Tag("acidity", "bright") in profile.tag_affinities


@pytest.mark.parametrize(
    "answers, message",
    [
        ({"nope": "x"}, "unknown question"),
        ({"style": "loud"}, "unknown option"),
        ({"style": ["bold", "bright"]}, "only one option"),
        ({"known_wines": {"demo-alder-creek-00": 1}}, "not in this question"),
        ({"known_wines": {"demo-kestrel-34": 5}}, "must be -1, 0 or 1"),
        ({"budget": "cheap"}, "unknown option"),
    ],
)
def test_bad_answers_rejected(spec, answers, message):
    with pytest.raises(ValueError, match=message):
        build_profile(spec, answers)


def test_later_feedback_overrides_quiz(spec):
    profile = build_profile(spec, {"known_wines": {"demo-kestrel-34": 1}})
    ctx = profile.to_user_context("u", disliked=["demo-kestrel-34"], liked=["demo-alder-creek-00"])
    assert ctx.liked == ("demo-alder-creek-00",)
    assert ctx.disliked == ("demo-kestrel-34",)


def test_spec_problems_flag_unknown_items_and_tags(demo_catalog):
    spec = QuizSpec.model_validate(
        {
            "version": "x",
            "questions": [
                {"id": "q", "type": "rate_items", "prompt": "?", "items": ["ghost"]},
                {
                    "id": "s",
                    "type": "choice",
                    "prompt": "?",
                    "options": [{"id": "o", "label": "O", "tags": {"varietal:unobtainium": 1}}],
                },
            ],
        }
    )
    problems = spec.problems(demo_catalog.items)
    assert "q: unknown item 'ghost'" in problems
    assert any("unobtainium" in p for p in problems)
