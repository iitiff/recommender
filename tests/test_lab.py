from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from recsys.lab.app import build_view, create_app
from recsys.lab.core import Lab
from tests.conftest import DEMO_DIR

HX = {"HX-Request": "true"}


@pytest.fixture(scope="module")
def lab():
    return Lab(DEMO_DIR, DEMO_DIR / "interactions.csv")


@pytest.fixture
def client(lab):
    return TestClient(create_app(lab))


def titles(html: str) -> list[str]:
    return re.findall(r'class="rec-title">([^<]+)<', html)


def act(client, action, value="") -> str:
    r = client.post("/act", data={"action": action, "value": value}, headers=HX)
    assert r.status_code == 200
    return r.text


def test_page_renders_user_experience_with_model_view_hidden(client):
    html = client.get("/").text
    assert "User experience" in html and "Lab controls" in html
    assert 'aria-label="Model view"' not in html
    assert len(titles(html)) == 5


def test_toggle_opens_overlay_with_explanations(client):
    client.get("/")
    html = act(client, "toggle_model")
    assert 'aria-label="Model view"' in html
    assert "How Hybrid (tag + popularity) works" in html
    assert "Candidate funnel" in html and "Offline scorecard" in html


def test_never_shows_quiz_only_items(client, lab):
    client.get("/")
    quiz_only = {i.title for i in lab.items if not i.recommendable}
    for persona in ("bold", "elegant", "spicy", "new"):
        assert not set(titles(act(client, "persona", persona))) & quiz_only


def test_switching_model_changes_list_and_explanation(client):
    client.get("/")
    act(client, "persona", "bold")
    hybrid = act(client, "model", "3")
    popular = act(client, "model", "1")
    assert "Because you liked" in hybrid
    assert "Popular with other visitors" in popular
    assert titles(hybrid) != titles(popular)


def test_cold_start_banner_for_new_visitor(client):
    client.get("/")
    act(client, "model", "2")
    html = act(client, "persona", "new")
    assert "We don&#39;t know your taste yet" in html


def test_feedback_reranks_and_logs_a_rating_event(client):
    client.get("/")
    act(client, "persona", "bold")
    act(client, "model", "2")
    html = act(client, "toggle_model")
    first = titles(html)[0]
    item_id = re.search(
        r'name="value" value="([^"]+)">\s*<button type="submit" class="thumb up"', html
    )
    html = act(client, "dislike", item_id.group(1))
    assert first not in titles(html)
    assert "rating" in html and "value=-1" in html
    assert "pushed down" in html  # the model-view note


def test_budget_filters_candidates(client):
    client.get("/")
    act(client, "persona", "bold")
    act(client, "toggle_model")
    html = act(client, "budget", "under_30")
    assert "Within budget" in html
    prices = [float(p) for p in re.findall(r"~\$(\d+)", html)]
    assert prices and all(p <= 30 for p in prices)


def test_live_quiz_builds_a_profile(client, lab):
    client.get("/")
    act(client, "model", "2")
    act(client, "persona", "quiz")
    html = act(client, "quiz_rate", "demo-copperline-38:1")
    assert 'aria-pressed="true">Liked it' in html
    assert "Because you liked" in html


def test_breakdown_matches_list_scores(lab):
    session = lab.new_session()
    for model_index in range(len(lab.models)):
        session.model = model_index
        run = lab.run(session)
        for rec in run.recs:
            b = run.model.score_breakdown(run.user, rec.item_id)
            assert b is not None
            assert b.total == pytest.approx(rec.score)


def test_view_without_interactions_says_not_measured():
    lab = Lab(DEMO_DIR)
    view = build_view(lab, lab.new_session())
    assert view["scorecard"][0]["ndcg"] == "not measured"


def test_plain_form_post_redirects_without_htmx(client):
    client.get("/")
    r = client.post("/act", data={"action": "toggle_model"}, follow_redirects=False)
    assert r.status_code == 303


def test_header_and_budget_controls_render_correctly(client):
    html = client.get("/").text
    assert "Catalog: 40 items · 30 recommendable" in html
    # Exactly one budget option is selected (a quiz option id once collided with the lab's own).
    budget_bar = html.split("2 · Their budget")[1].split("3 · Serve them")[0]
    assert budget_bar.count('aria-pressed="true"') == 1


def test_display_capitalizes_without_breaking_apostrophes():
    from recsys.lab.core import display

    assert display("frog's leap") == "Frog's Leap"
    assert display("petite sirah") == "Petite Sirah"
