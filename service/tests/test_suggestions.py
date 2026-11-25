import asyncio
from datetime import datetime, timedelta, timezone

from app.immich import AssetInfo
from app.jobs import JobStore
from app.settings_store import AppSettings, SettingsStore
from app.suggestions import SuggestionStore, due, group_events, ideas_for, scan

T0 = datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)


def a(i: int, minutes: float, lat=None, lon=None, video=False, people=(), city=None, w=4000, h=3000, name=None) -> AssetInfo:
    return AssetInfo(
        id=f"{i:08d}-0000-4000-8000-000000000000", type="VIDEO" if video else "IMAGE", filename=name or f"IMG_{i}.jpg",
        mime="image/jpeg", width=w, height=h, taken_at=T0 + timedelta(minutes=minutes), is_favorite=False,
        visibility="timeline", is_trashed=False, latitude=lat, longitude=lon, people=tuple(people),
        duration_ms=8000 if video else None, make="Google", city=city,
    )


def test_events_split_on_time_gaps_and_distance():
    assets = [a(0, 0), a(1, 30), a(2, 60)] + [a(3, 60 + 4 * 60 + 1)]  # 4h gap -> new event
    assert [len(e.items) for e in group_events(assets, 3)] == [3, 1]
    # Same afternoon, 60 km apart -> two events.
    moved = [a(0, 0, 11.78, 78.21), a(1, 20, 11.78, 78.21), a(2, 40, 11.65, 78.15 + 0.6)]
    assert [len(e.items) for e in group_events(moved, 3)] == [2, 1]


def test_ideas_follow_the_rules():
    s = AppSettings()
    trip = group_events([a(i, i * 10, people=["Asha"] if i == 3 else (), city="Yercaud") for i in range(16)] + [a(99, 165, video=True)], 3)[0]
    kinds = {i.kind for i in ideas_for(trip, s)}
    assert kinds == {"collage", "highlight", "cinematic", "popout"}
    idea = next(i for i in ideas_for(trip, s) if i.kind == "collage")
    assert idea.title.startswith("Yercaud") and idea.params["event_from"] < idea.params["event_to"]
    assert idea.params["caption"] == "Yercaud"
    # Too few photos: nothing.
    small = group_events([a(i, i) for i in range(3)], 3)[0]
    assert ideas_for(small, s) == []
    # Screenshots don't count towards a collage.
    shots = group_events([a(i, i, name=f"Screenshot_{i}.png") for i in range(10)], 3)[0]
    assert "collage" not in {i.kind for i in ideas_for(shots, s)}


def test_cinematic_needs_people_and_size():
    s = AppSettings()
    ev = group_events([a(i, i, people=["Ravi"], w=800, h=600) for i in range(6)], 3)[0]
    assert "cinematic" not in {i.kind for i in ideas_for(ev, s)}


class FakeGateway:
    def __init__(self, assets, people=(), ocr=None):
        self.assets = assets
        self.people = list(people)
        self.ocr = ocr or {}
        self.calls = []

    async def search(self, **kw):
        self.calls.append(kw)
        return [x for x in self.assets if kw["taken_after"] <= x.taken_at <= kw["taken_before"]]

    async def people_with_birthdays(self):
        return self.people

    async def ocr_hits(self, text, a, b):
        return self.ocr.get(text, 0)


def test_scan_adds_each_idea_once_and_respects_daily_cap(db):
    s = SettingsStore(db).get()
    s.suggestions.max_per_day = 2
    store, jobs = SuggestionStore(db), JobStore(db)
    gw = FakeGateway([a(i, i * 10, people=["Asha"]) for i in range(20)])
    now = T0 + timedelta(days=1)
    res = asyncio.run(scan(gw, store, jobs, s, now=now))
    assert res["added"] == 2 and len(store.list()) == 2
    assert gw.calls[0]["taken_after"] == now - timedelta(days=30)  # first scan looks back 30 days
    # Next scan only looks at photos since the last one, and never repeats an idea.
    res2 = asyncio.run(scan(gw, store, jobs, s, now=now + timedelta(days=1)))
    assert gw.calls[1]["taken_after"] == now and res2["added"] == 0


def test_auto_mode_queues_jobs(db):
    s = AppSettings()
    s.suggestions.mode = "auto"
    store, jobs = SuggestionStore(db), JobStore(db)
    asyncio.run(scan(FakeGateway([a(i, i * 10) for i in range(8)]), store, jobs, s, now=T0 + timedelta(days=1)))
    assert store.list() == [] and len(jobs.queued()) == 1 and jobs.queued()[0].origin == "suggestion"


def test_due_once_a_day_after_scan_time():
    s = AppSettings()
    s.suggestions.scan_at = "03:00"
    local = datetime(2026, 11, 24, 3, 30).astimezone()
    assert due(s, None, local)
    assert not due(s, local, local + timedelta(minutes=5))
    assert due(s, local, local + timedelta(days=1))
    assert not due(s, None, local.replace(hour=2))
    s.suggestions.mode = "off"
    assert not due(s, None, local)


def test_rank_prefers_big_events_and_mixes_kinds():
    from app.suggestions import Idea, rank

    def idea(k, kind, score):
        return Idea(k, kind, k, "", {}, [], score)

    ideas = [idea(f"c{i}", "cinematic", 1.0 + i * 0.01) for i in range(5)] + [idea("h", "highlight", 0.9), idea("col", "collage", 0.8)]
    picked = [i.kind for i in rank(ideas, 3)]
    assert picked.count("cinematic") <= 2 and "highlight" in picked
    assert [i.kind for i in rank([idea("a", "cinematic", 1), idea("b", "cinematic", 2)], 3)] == ["cinematic", "cinematic"]


def test_scan_ranks_across_events_not_first_come(db):
    s = AppSettings()
    s.suggestions.max_per_day = 3
    store, jobs = SuggestionStore(db), JobStore(db)
    small = [a(i, i, people=["Asha"]) for i in range(3)]  # early: only a cinematic idea
    big = [a(100 + i, 2000 + i * 5, people=["Ravi"] if i % 4 == 0 else ()) for i in range(30)]  # later, big trip
    asyncio.run(scan(FakeGateway(small + big), store, jobs, s, now=T0 + timedelta(days=3)))
    kinds = {x["kind"] for x in store.list()}
    assert {"collage", "highlight"} <= kinds
