"""Rec Lab: run real models on a catalog and explain what they do.

Three zones, mirroring the agreed mockup:
- Lab controls: who is visiting (persona or live quiz), their budget, which model.
- User experience: exactly what the visitor would see.
- Model view (an overlay): why the model produced that list.

Everything goes through the production code paths (quiz → profile, candidate
filters, Recommender.recommend, explanations, the event schema), so the lab
shows the real behaviour, not a simulation. Two deliberate differences:
- Events go to an in-memory log shown on screen, never the database, so
  experimenting can't pollute training data.
- Models are fitted once at startup. Lab 👍/👎 change the visitor's profile
  instantly (that's in-session personalization); retraining is a separate,
  scheduled job in production.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from recsys.candidates import CandidateFilter, CandidateResult, filter_candidates, price_of
from recsys.catalog import load_catalog
from recsys.core.types import (
    Explanation,
    ExplanationKind,
    Item,
    ItemId,
    Offer,
    ScoredItem,
    Tag,
    TrainingData,
    UserContext,
)
from recsys.eval import evaluate, load_interactions_csv, temporal_split
from recsys.events.schema import Event, EventType, SlateItem
from recsys.models import REGISTRY, Recommender, build_model_entries
from recsys.profile.quiz import (
    BudgetQuestion,
    ChoiceQuestion,
    QuizProfile,
    QuizSpec,
    RateItemsQuestion,
    build_profile,
)

K = 5
TENANT = "lab"


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    blurb: str = ""
    answers: dict[str, Any] = Field(default_factory=dict)  # same shape as build_profile()
    interactive: bool = False  # the visitor answers the quiz live in the lab


class LabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = "Rec Lab"
    meta_facets: list[str] = Field(default_factory=list)  # shown under each item title
    models: list[dict[str, Any]] = Field(default_factory=list)  # like models.yaml entries
    default_model: int = 0
    personas: list[Persona] = Field(
        default_factory=lambda: [Persona(id="new", label="New visitor", blurb="No signal yet.")]
    )

    @classmethod
    def load(cls, path: Path) -> LabConfig:
        if not path.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


@dataclass
class Session:
    persona: str
    model: int
    budget: str | None = None  # budget option id; None = the persona's own answer
    quiz: dict[str, Any] = field(default_factory=dict)  # live answers (interactive persona)
    likes: list[ItemId] = field(default_factory=list)
    dislikes: list[ItemId] = field(default_factory=list)
    selected: ItemId | None = None
    show_model: bool = False
    events: list[Event] = field(default_factory=list)
    note: str = ""
    seq: int = 0


ANY_BUDGET = "_no_limit"  # lab-only marker; must not collide with quiz option ids


class Lab:
    def __init__(self, catalog_dir: Path, interactions: Path | None = None) -> None:
        catalog_dir = Path(catalog_dir)
        catalog = load_catalog(catalog_dir)
        self.items: list[Item] = [i for i in catalog.items if i.is_active]
        self.by_id: dict[ItemId, Item] = {i.item_id: i for i in self.items}
        self.offers: list[Offer] = catalog.offers
        self.config = LabConfig.load(catalog_dir / "lab.yaml")

        quiz_path = catalog_dir / "quiz.yaml"
        self.quiz: QuizSpec | None = QuizSpec.from_yaml(quiz_path) if quiz_path.exists() else None
        if self.quiz is not None:
            problems = self.quiz.problems(self.items)
            if problems:
                raise ValueError("quiz.yaml doesn't match the catalog:\n  " + "\n  ".join(problems))

        entries = self.config.models or [{"model": name} for name in REGISTRY]
        self.models: list[tuple[str, Recommender]] = build_model_entries(entries)
        self.interactions = load_interactions_csv(interactions) if interactions else []
        data = TrainingData(self.items, self.interactions)
        for _, model in self.models:
            model.fit(data)

        # Facets that are bins of numeric attributes, e.g. body:full.
        self.numeric_attrs = sorted({a for i in self.items for a in i.attributes} - {"price"})
        self.attr_max = {
            a: max(i.attributes[a] for i in self.items if a in i.attributes)
            for a in self.numeric_attrs
        }
        self.usable_offers: dict[ItemId, list[Offer]] = defaultdict(list)
        for o in self.offers:
            if o.in_stock:
                self.usable_offers[o.item_id].append(o)
        self.scorecard = self._scorecard(entries) if self.interactions else {}

    # -- setup helpers --------------------------------------------------------

    def _scorecard(self, entries: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        """Offline metrics for each lab model, via the real eval harness.

        Fresh model instances: evaluate() refits on the training split, and the
        lab's own models must stay fitted on all the data.
        """
        split = temporal_split(self.items, self.interactions)
        fresh = [model for _, model in build_model_entries(entries)]
        report = evaluate(fresh, split, ks=(K, 10), baseline=None, n_resamples=500)
        card = {}
        for result in report.results:
            ndcg = result.metrics["ndcg@10"]
            card[result.model_id] = {
                "ndcg": f"{ndcg.mean:.3f} [{ndcg.low:.3f}–{ndcg.high:.3f}]",
                "coverage": f"{result.aggregates['coverage@10']:.2f}",
                "popularity": f"{result.metrics['avg_popularity@10'].mean:.3f}",
            }
        card["_info"] = {
            "users": str(report.info["users_evaluated"]),
            "fingerprint": str(report.info["fingerprint"]),
        }
        return card

    def new_session(self) -> Session:
        default = min(self.config.default_model, len(self.models) - 1)
        return Session(persona=self.config.personas[0].id, model=default)

    def persona(self, session: Session) -> Persona:
        return next(p for p in self.config.personas if p.id == session.persona)

    def _question(self, kind: type) -> Any:
        if self.quiz is None:
            return None
        return next((q for q in self.quiz.questions if isinstance(q, kind)), None)

    @property
    def rate_question(self) -> RateItemsQuestion | None:
        return self._question(RateItemsQuestion)

    @property
    def budget_question(self) -> BudgetQuestion | None:
        return self._question(BudgetQuestion)

    @property
    def choice_questions(self) -> list[ChoiceQuestion]:
        if self.quiz is None:
            return []
        return [q for q in self.quiz.questions if isinstance(q, ChoiceQuestion)]

    # -- the pipeline ---------------------------------------------------------

    def answers(self, session: Session) -> dict[str, Any]:
        persona = self.persona(session)
        answers: dict[str, Any] = {
            k: (dict(v) if isinstance(v, dict) else v) for k, v in persona.answers.items()
        }
        if persona.interactive:
            for key, value in session.quiz.items():
                if isinstance(value, dict):
                    answers.setdefault(key, {}).update(value)
                else:
                    answers[key] = value
        bq = self.budget_question
        if bq is not None and session.budget is not None:
            if session.budget == ANY_BUDGET:
                answers.pop(bq.id, None)
            else:
                answers[bq.id] = session.budget
        return answers

    def profile(self, session: Session) -> QuizProfile:
        if self.quiz is None:
            return QuizProfile(quiz_version="none")
        return build_profile(self.quiz, self.answers(session))

    def run(self, session: Session) -> Run:
        label, model = self.models[session.model]
        profile = self.profile(session)
        user = profile.to_user_context(
            f"lab-{session.persona}", liked=session.likes, disliked=session.dislikes
        )
        candidates = filter_candidates(
            self.items,
            self.offers,
            CandidateFilter(min_price=profile.min_price, max_price=profile.max_price),
        )
        recs = model.recommend(user, k=K, candidates=candidates.item_ids)
        return Run(label, model, profile, user, candidates, recs)

    def price(self, item_id: ItemId) -> float | None:
        return price_of(self.by_id[item_id], self.usable_offers.get(item_id, []))

    # -- actions --------------------------------------------------------------

    def act(self, session: Session, action: str, value: str = "") -> None:
        """Apply one lab action. Unknown values are ignored (stale buttons)."""
        reruns = True
        if action == "persona" and any(p.id == value for p in self.config.personas):
            session.persona, session.budget = value, None
            session.likes, session.dislikes, session.quiz = [], [], {}
            session.selected, session.note = None, ""
        elif action == "model" and value.isdigit() and int(value) < len(self.models):
            session.model, session.selected, session.note = int(value), None, ""
        elif action == "budget":
            session.budget, session.selected = value or None, None
        elif action == "quiz_rate" and self.rate_question is not None:
            item_id, _, rating = value.partition(":")
            if item_id in self.rate_question.items and rating in ("1", "-1", "0"):
                session.quiz.setdefault(self.rate_question.id, {})[item_id] = int(rating)
                session.selected = None
        elif action == "quiz_choice":
            question_id, _, option_id = value.partition(":")
            for q in self.choice_questions:
                if q.id == question_id and any(o.id == option_id for o in q.options):
                    session.quiz[q.id] = option_id
                    session.selected = None
        elif action in ("like", "dislike") and value in self.by_id:
            self._feedback(session, value, 1 if action == "like" else -1)
        elif action == "select" and value in self.by_id:
            session.selected, session.show_model, reruns = value, True, False
        elif action == "toggle_model":
            session.show_model, reruns = not session.show_model, False
        elif action == "reset":
            session.likes, session.dislikes, session.selected, session.note = [], [], None, ""
            session.events = []
        else:
            reruns = False
        if reruns:
            self._log_slate(session)

    def _feedback(self, session: Session, item_id: ItemId, value: int) -> None:
        session.likes = [i for i in session.likes if i != item_id]
        session.dislikes = [i for i in session.dislikes if i != item_id]
        (session.likes if value > 0 else session.dislikes).append(item_id)
        session.selected = None
        _, model = self.models[session.model]
        title = self.by_id[item_id].title
        if model.personalized:
            effect = (
                "It's now an anchor: similar wines move up and it can appear in "
                '"because you liked…".'
                if value > 0
                else "Wines similar to it are pushed down."
            )
        else:
            effect = (
                f"{model.name} ignores your feedback: the list only changes because "
                "this item is removed."
            )
        session.note = f"You {'liked' if value > 0 else 'disliked'} {title}. {effect}"
        self._log(
            session,
            Event(
                event_type=EventType.RATING,
                tenant_id=TENANT,
                user_id=f"lab-{session.persona}",
                item_id=item_id,
                request_id=f"lab-{session.seq}",
                surface="lab",
                properties={"value": value},
            ),
        )

    def _log_slate(self, session: Session) -> None:
        run = self.run(session)
        session.seq += 1
        self._log(
            session,
            Event(
                event_type=EventType.SLATE_SERVED,
                tenant_id=TENANT,
                user_id=run.user.user_id,
                request_id=f"lab-{session.seq}",
                surface="lab",
                model_id=run.model.model_id,
                properties={
                    "items": [
                        SlateItem(
                            item_id=r.item_id,
                            position=n,
                            score=r.score,
                            reason=r.explanation.kind if r.explanation else None,
                        ).model_dump(mode="json")
                        for n, r in enumerate(run.recs)
                    ],
                    "k": K,
                    "candidate_count": len(run.candidates.item_ids - run.user.excluded),
                },
            ),
        )

    @staticmethod
    def _log(session: Session, event: Event) -> None:
        session.events = [event, *session.events][:8]


@dataclass(frozen=True)
class Run:
    label: str
    model: Recommender
    profile: QuizProfile
    user: UserContext
    candidates: CandidateResult
    recs: list[ScoredItem]

    @property
    def cold(self) -> bool:
        return self.model.personalized and not self.user.liked and not self.user.tag_affinities


# -- presentation helpers (pure) ------------------------------------------------


def display(value: str) -> str:
    """Capitalize words for display ("frog's leap" -> "Frog's Leap", unlike str.title)."""
    return " ".join(w[:1].upper() + w[1:] for w in value.split())


def pretty_tag(tag: Tag, numeric_attrs: list[str]) -> str:
    return f"{tag.facet}: {tag.value}" if tag.facet in numeric_attrs else tag.value


def explanation_text(
    exp: Explanation | None, model: Recommender, lab: Lab, cold: bool
) -> tuple[str, str]:
    """(why, detail) as a visitor would read them."""
    if exp is None:
        return (
            ("Random pick", "Baseline: no signal used")
            if model.name == "random"
            else (
                "Recommended for you",
                "",
            )
        )
    tags = ", ".join(pretty_tag(t, lab.numeric_attrs) for t in exp.shared_tags)
    if exp.kind == ExplanationKind.BECAUSE_YOU_LIKED and exp.anchor_item_ids:
        anchor = lab.by_id.get(exp.anchor_item_ids[0])
        return (
            f"Because you liked {anchor.title if anchor else exp.anchor_item_ids[0]}",
            f"Both: {tags}" if tags else "Similar overall",
        )
    if exp.kind == ExplanationKind.MATCHES_YOUR_TASTE:
        return "Matches your taste", f"You said you like: {tags}" if tags else ""
    if exp.kind == ExplanationKind.POPULAR:
        detail = "We don't know your taste yet" if cold else "Not personalized"
        return "Popular with other visitors", detail
    if exp.kind == ExplanationKind.EXPLORATION:
        detail = "We don't know your taste yet" if cold else "Not personalized"
        return "A good place to start", detail
    if exp.kind == ExplanationKind.RECENTLY_VIEWED:
        return "You looked at this recently", ""
    if exp.kind == ExplanationKind.ALSO_VIEWED and exp.anchor_item_ids:
        anchor = lab.by_id.get(exp.anchor_item_ids[0])
        return (
            f"Shoppers who viewed {anchor.title if anchor else exp.anchor_item_ids[0]} "
            "also looked at this",
            "",
        )
    if exp.kind == ExplanationKind.TRENDING:
        return "Trending now", "Not personalized"
    if exp.kind == ExplanationKind.BUY_AGAIN:
        return "Buy it again", "You bought this before"
    if exp.kind == ExplanationKind.ALTERNATIVE and exp.anchor_item_ids:
        anchor = lab.by_id.get(exp.anchor_item_ids[0])
        return (
            f"An alternative to {anchor.title if anchor else exp.anchor_item_ids[0]}",
            "In your cart",
        )
    if exp.kind == ExplanationKind.NEW_FOR_YOU:
        anchor = lab.by_id.get(exp.anchor_item_ids[0]) if exp.anchor_item_ids else None
        detail = f"Shoppers who liked {anchor.title} often try this next" if anchor else ""
        return "Something new for you", detail
    return "Recommended for you", tags
