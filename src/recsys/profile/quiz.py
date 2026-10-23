"""Taste quiz: the cold-start solution.

A quiz is configuration (quiz.yaml in a catalog directory), not code, so each
site (or store) writes its own. Three question types:

- `rate_items`: "Have you had any of these?" Answers become real likes and
  dislikes on those items, the strongest cold-start signal because they're
  the same kind of data as later feedback. Quiz items are usually
  well-known, non-recommendable items: recognition beats prestige.
- `choice`: style questions ("bold or light?"). Each option maps to tag
  affinities, e.g. {"body:full": 1.0, "oak:oaky": 0.5}.
- `budget`: a price band, applied as a hard candidate filter (never a score).

The raw answers are what get stored (`quiz_responses`); the profile is derived
and can be recomputed whenever the mapping changes.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from recsys.catalog.loader import normalize
from recsys.core.types import Item, ItemId, Tag, UserContext


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RateItemsQuestion(_Model):
    id: str
    type: Literal["rate_items"]
    prompt: str
    items: list[str]


class ChoiceOption(_Model):
    id: str
    label: str
    tags: dict[str, float] = Field(default_factory=dict)  # "facet:value" -> affinity


class ChoiceQuestion(_Model):
    id: str
    type: Literal["choice"]
    prompt: str
    options: list[ChoiceOption]
    multi: bool = False


class BudgetOption(_Model):
    id: str
    label: str
    min_price: float | None = None
    max_price: float | None = None


class BudgetQuestion(_Model):
    id: str
    type: Literal["budget"]
    prompt: str
    options: list[BudgetOption]


Question = Annotated[
    RateItemsQuestion | ChoiceQuestion | BudgetQuestion, Field(discriminator="type")
]


class QuizSpec(_Model):
    version: str
    questions: list[Question]

    @classmethod
    def from_yaml(cls, path: Path) -> QuizSpec:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})

    def question(self, question_id: str) -> RateItemsQuestion | ChoiceQuestion | BudgetQuestion:
        for q in self.questions:
            if q.id == question_id:
                return q
        raise ValueError(f"unknown question {question_id!r}")

    def problems(self, items: Sequence[Item]) -> list[str]:
        """Check the quiz against a catalog: every item and tag must exist."""
        ids = {i.item_id for i in items}
        tags = {t for i in items for t in i.tags}
        found: list[str] = []
        seen_ids: set[str] = set()
        for q in self.questions:
            if q.id in seen_ids:
                found.append(f"duplicate question id {q.id!r}")
            seen_ids.add(q.id)
            if isinstance(q, RateItemsQuestion):
                found += [f"{q.id}: unknown item {i!r}" for i in q.items if i not in ids]
            elif isinstance(q, ChoiceQuestion):
                for option in q.options:
                    for raw in option.tags:
                        if _tag(raw) not in tags:
                            found.append(f"{q.id}/{option.id}: no item has tag {raw!r}")
        return found


def _tag(raw: str) -> Tag:
    tag = Tag.parse(raw)
    return Tag(tag.facet, normalize(tag.value))


@dataclass(frozen=True)
class QuizProfile:
    quiz_version: str
    liked: tuple[ItemId, ...] = ()
    disliked: tuple[ItemId, ...] = ()
    tag_affinities: Mapping[Tag, float] = field(default_factory=dict)
    min_price: float | None = None
    max_price: float | None = None

    def to_user_context(
        self,
        user_id: str | None = None,
        *,
        liked: Collection[ItemId] = (),
        disliked: Collection[ItemId] = (),
    ) -> UserContext:
        """Merge quiz answers with later on-site feedback (which wins on conflict)."""
        later_likes, later_dislikes = set(liked), set(disliked)
        all_liked = (set(self.liked) - later_dislikes) | later_likes
        all_disliked = (set(self.disliked) - later_likes) | later_dislikes
        return UserContext(
            user_id=user_id,
            tag_affinities=dict(self.tag_affinities),
            liked=tuple(sorted(all_liked)),
            disliked=tuple(sorted(all_disliked)),
        )


def build_profile(spec: QuizSpec, answers: Mapping[str, Any]) -> QuizProfile:
    """Turn raw answers into a profile.

    answers = {
        "known_wines": {"daou": 1, "meiomi": -1, "bogle": 0},  # rate_items: +1 / -1 / 0 (not had)
        "style": "bold",                                        # choice (list if multi)
        "budget": "under_50",                                   # budget
    }
    Unanswered questions are fine (quizzes get abandoned); unknown ids are errors.
    """
    liked: list[ItemId] = []
    disliked: list[ItemId] = []
    affinities: dict[Tag, float] = {}
    min_price = max_price = None

    for question_id, answer in answers.items():
        q = spec.question(question_id)
        if isinstance(q, RateItemsQuestion):
            if not isinstance(answer, Mapping):
                raise ValueError(f"{q.id}: expected {{item_id: +1/-1/0}}")
            for item_id, value in answer.items():
                if item_id not in q.items:
                    raise ValueError(f"{q.id}: {item_id!r} is not in this question")
                if value not in (-1, 0, 1):
                    raise ValueError(f"{q.id}: rating must be -1, 0 or 1, got {value!r}")
                if value == 1:
                    liked.append(item_id)
                elif value == -1:
                    disliked.append(item_id)
        elif isinstance(q, ChoiceQuestion):
            chosen = answer if isinstance(answer, list) else [answer]
            if len(chosen) > 1 and not q.multi:
                raise ValueError(f"{q.id}: only one option allowed")
            options = {o.id: o for o in q.options}
            for option_id in chosen:
                if option_id not in options:
                    raise ValueError(f"{q.id}: unknown option {option_id!r}")
                for raw, weight in options[option_id].tags.items():
                    tag = _tag(raw)
                    affinities[tag] = affinities.get(tag, 0.0) + weight
        else:
            options = {o.id: o for o in q.options}
            if answer not in options:
                raise ValueError(f"{q.id}: unknown option {answer!r}")
            min_price, max_price = options[answer].min_price, options[answer].max_price

    return QuizProfile(
        quiz_version=spec.version,
        liked=tuple(liked),
        disliked=tuple(disliked),
        tag_affinities=affinities,
        min_price=min_price,
        max_price=max_price,
    )
