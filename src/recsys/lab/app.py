"""FastAPI app for the Rec Lab: `recsys lab CATALOG_DIR [--interactions CSV]`.

Server-rendered with Jinja2. Every button is a real <form> POST, so the lab
works without JavaScript; HTMX (when loaded) swaps just the #lab block so the
page doesn't flash. A local learning tool: sessions live in memory.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recsys.lab.core import ANY_BUDGET, K, Lab, Session, display, explanation_text, pretty_tag

HERE = Path(__file__).parent
COOKIE = "reclab"


def build_view(lab: Lab, session: Session) -> dict[str, Any]:
    """Everything the template needs, as plain data."""
    run = lab.run(session)
    persona = lab.persona(session)
    model = run.model

    recs = []
    for rank, rec in enumerate(run.recs, start=1):
        item = lab.by_id[rec.item_id]
        why, detail = explanation_text(rec.explanation, model, lab, run.cold)
        price = lab.price(rec.item_id)
        meta = [
            ", ".join(sorted(display(t.value) for t in item.tags if t.facet == facet))
            for facet in lab.config.meta_facets
        ]
        if price is not None:
            meta.append(f"~${price:.0f}")
        recs.append(
            {
                "rank": rank,
                "id": rec.item_id,
                "title": item.title,
                "meta": " · ".join(m for m in meta if m),
                "why": why,
                "detail": detail,
                "selected": rec.item_id == session.selected,
            }
        )
    selected_id = session.selected if any(r["id"] == session.selected for r in recs) else None
    if selected_id is None and recs:
        selected_id = recs[0]["id"]

    breakdown = None
    if selected_id is not None:
        b = model.score_breakdown(run.user, selected_id)
        if b is not None:
            biggest = max([abs(c.value) for c in b.components] + [1e-9])
            breakdown = {
                "title": lab.by_id[selected_id].title,
                "rank": next(r["rank"] for r in recs if r["id"] == selected_id),
                "total": b.total,
                "formula": b.formula,
                "components": [
                    {
                        "label": c.label,
                        "value": c.value,
                        "width": round(100 * abs(c.value) / max(biggest, 1.0), 1),
                        "negative": c.value < 0,
                    }
                    for c in b.components
                ],
                "anchors": [
                    {"title": lab.by_id[a].title, "sim": s, "best": n == 0}
                    for n, (a, s) in enumerate(b.anchors)
                ],
            }

    signals = [
        {
            "title": lab.by_id[i].title,
            "kind": "👍",
            "source": "lab" if i in session.likes else "quiz",
            "positive": True,
        }
        for i in run.user.liked
    ] + [
        {
            "title": lab.by_id[i].title,
            "kind": "👎",
            "source": "lab" if i in session.dislikes else "quiz",
            "positive": False,
        }
        for i in run.user.disliked
    ]
    liked_items = [lab.by_id[i] for i in run.user.liked if i in lab.by_id]
    taste_bars = []
    for attr in lab.numeric_attrs:
        values = [i.attributes[attr] for i in liked_items if attr in i.attributes]
        if values:
            avg = sum(values) / len(values)
            taste_bars.append(
                {
                    "label": attr,
                    "value": avg,
                    "width": round(100 * avg / lab.attr_max[attr], 1),
                }
            )
    affinities = sorted(
        ((t, w) for t, w in run.user.tag_affinities.items() if w > 0), key=lambda x: -x[1]
    )

    unrated = len(run.candidates.item_ids - run.user.excluded)
    total = run.candidates.steps[0][1] or 1
    funnel = [
        {"label": label, "n": n, "width": round(100 * n / total, 1), "model": False}
        for label, n in run.candidates.steps
    ]
    funnel.append(
        {"label": "Not already rated", "n": unrated, "width": round(100 * unrated / total, 1)}
    )
    funnel.append(
        {
            "label": f"Ranked by {run.label} → shown",
            "n": len(recs),
            "width": round(100 * len(recs) / total, 1),
            "model": True,
        }
    )

    bq = lab.budget_question
    budgets = []
    if bq is not None:
        chosen = lab.answers(session).get(bq.id, ANY_BUDGET)
        budgets = [{"id": o.id, "label": o.label, "on": o.id == chosen} for o in bq.options]
        budgets.append({"id": ANY_BUDGET, "label": "No limit", "on": chosen == ANY_BUDGET})

    quiz_rows, choice_questions = [], []
    if persona.interactive and lab.rate_question is not None:
        answered = lab.answers(session).get(lab.rate_question.id, {})
        for item_id in lab.rate_question.items:
            item = lab.by_id[item_id]
            meta = [
                ", ".join(sorted(display(t.value) for t in item.tags if t.facet == facet))
                for facet in lab.config.meta_facets
            ]
            quiz_rows.append(
                {
                    "id": item_id,
                    "title": item.title,
                    "meta": " · ".join(m for m in meta if m),
                    "value": answered.get(item_id, 0),
                }
            )
    if persona.interactive:
        answers = lab.answers(session)
        for q in lab.choice_questions:
            choice_questions.append(
                {
                    "id": q.id,
                    "prompt": q.prompt,
                    "options": [
                        {"id": o.id, "label": o.label, "on": answers.get(q.id) == o.id}
                        for o in q.options
                    ],
                }
            )

    scorecard = []
    for n, (label, m) in enumerate(lab.models):
        row = lab.scorecard.get(m.model_id)
        scorecard.append(
            {
                "label": label,
                "model_id": m.model_id,
                "current": n == session.model,
                "ndcg": row["ndcg"] if row else "not measured",
                "coverage": row["coverage"] if row else "—",
                "popularity": row["popularity"] if row else "—",
            }
        )

    events = []
    for e in session.events:
        if e.event_type == "slate_served":
            ids = ", ".join(i["item_id"] for i in e.properties["items"])
            text = f"slate_served  {e.request_id}  model_id={e.model_id}  items=[{ids}]"
        else:
            value = e.properties.get("value", "")
            text = f"{e.event_type:<13} {e.request_id}  item={e.item_id}  value={value}"
        events.append({"text": text, "type": str(e.event_type)})

    return {
        "title": lab.config.title,
        "k": K,
        "personas": [
            {"id": p.id, "label": p.label, "on": p.id == session.persona}
            for p in lab.config.personas
        ],
        "persona": persona,
        "models": [
            {"index": n, "label": label, "on": n == session.model}
            for n, (label, _) in enumerate(lab.models)
        ],
        "model": model,
        "model_label": run.label,
        "budgets": budgets,
        "budget_label": next((b["label"] for b in budgets if b["on"]), ""),
        "feedback_count": len(session.likes) + len(session.dislikes),
        "show_model": session.show_model,
        "quiz_rows": quiz_rows,
        "choice_questions": choice_questions,
        "cold": run.cold,
        "recs": recs,
        "selected_id": selected_id,
        "breakdown": breakdown,
        "signals": signals,
        "taste_bars": taste_bars,
        "affinities": [
            {"label": pretty_tag(t, lab.numeric_attrs), "weight": w} for t, w in affinities[:6]
        ],
        "funnel": funnel,
        "unknown_price": run.candidates.unknown_price,
        "scorecard": scorecard,
        "scorecard_info": lab.scorecard.get("_info"),
        "events": events,
        "note": session.note,
        "catalog_counts": {
            "total": len(lab.items),
            "recommendable": sum(1 for i in lab.items if i.recommendable),
        },
    }


def create_app(lab: Lab) -> FastAPI:
    app = FastAPI(title="Rec Lab")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    sessions: dict[str, Session] = {}

    def session_for(request: Request) -> tuple[str, Session]:
        sid = request.cookies.get(COOKIE)
        if sid not in sessions:
            sid = uuid.uuid4().hex
            sessions[sid] = lab.new_session()
            lab.act(sessions[sid], "reset")  # logs the first slate
        return sid, sessions[sid]

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        sid, session = session_for(request)
        response = templates.TemplateResponse(request, "lab.html", {"v": build_view(lab, session)})
        response.set_cookie(COOKIE, sid, httponly=True, samesite="lax")
        return response

    @app.post("/act")
    def act(request: Request, action: str = Form(...), value: str = Form("")) -> Response:
        sid, session = session_for(request)
        lab.act(session, action, value)
        if request.headers.get("HX-Request"):
            response: Response = templates.TemplateResponse(
                request, "_lab.html", {"v": build_view(lab, session)}
            )
        else:
            response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE, sid, httponly=True, samesite="lax")
        return response

    return app
