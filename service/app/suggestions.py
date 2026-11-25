"""Daily suggestions. Rules: docs/research/immich-vs-google-photos.md sections 3-5.

The scan looks at photos taken since the last scan (first scan: the last 30 days), groups them
into events, works out what kind of event each one is (app/events.py: birthday, festival, school
function, wedding, get-together, outing, trip) and proposes, per event:

- a **collage** when it has at least `min_good_photos` usable photos;
- a **highlight video** when it has 15+ items (or any clips and 10+ items);
- a **pop-out photo** of its best photo of 1-6 people (styles from Settings);
- a **cinematic photo** of a sharp single-subject portrait;
- once a year, before a named person's birthday: "<name> through the years" (highlight).

Events: no gap longer than `event_gap_hours`, split again where GPS jumps more than ~25 km;
consecutive days in the same city are merged back into one trip. Each idea is offered once, ever.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from .db import Database
from .events import OCR_HINTS, EventKind, Person, classify, home_location
from .immich import AssetInfo, ImmichError, ImmichGateway
from .imaging.quality import looks_like_screenshot
from .jobs import JobStore, now_iso
from .settings_store import AppSettings

log = logging.getLogger(__name__)
FIRST_SCAN_DAYS = 30
SPLIT_KM = 25.0
HIGHLIGHT_MIN_ITEMS = 15
YEARS_AHEAD_DAYS = 14  # "through the years" is offered in the two weeks before a birthday
OCR_MIN_ITEMS = 6


@dataclass
class Event:
    items: list[AssetInfo] = field(default_factory=list)
    kind: EventKind | None = None

    @property
    def start(self) -> datetime:
        return self.items[0].taken_at

    @property
    def end(self) -> datetime:
        return self.items[-1].taken_at

    @property
    def key(self) -> str:
        return f"{self.start:%Y%m%dT%H%M}-{len(self.items)}"

    @property
    def photos(self) -> list[AssetInfo]:
        return [a for a in self.items if a.type == "IMAGE"]

    @property
    def videos(self) -> list[AssetInfo]:
        return [a for a in self.items if a.type == "VIDEO"]

    @property
    def city(self) -> str | None:
        places = [a.city for a in self.items if a.city]
        return max(set(places), key=places.count) if places else None

    def title(self) -> str:
        if self.kind:
            return self.kind.title
        a, b = self.start, self.end
        if a.date() == b.date():
            when = f"{a.day} {a:%b %Y}"
        elif (a.year, a.month) == (b.year, b.month):
            when = f"{a.day}\u2013{b.day} {b:%b %Y}"
        else:
            when = f"{a.day} {a:%b} \u2013 {b.day} {b:%b %Y}"
        return f"{self.city} \u00b7 {when}" if self.city else when


def _km(a: AssetInfo, b: AssetInfo) -> float | None:
    if None in (a.latitude, a.longitude, b.latitude, b.longitude):
        return None
    p1, p2 = math.radians(a.latitude), math.radians(b.latitude)
    dp, dl = p2 - p1, math.radians(b.longitude - a.longitude)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def group_events(assets: list[AssetInfo], gap_hours: float) -> list[Event]:
    """Split time-ordered assets on long gaps and big location jumps, then merge consecutive
    days spent in the same city (a 3-day trip is one event, not three)."""
    events: list[Event] = []
    last_located: AssetInfo | None = None
    for a in sorted(assets, key=lambda x: x.taken_at):
        new = not events or (a.taken_at - events[-1].end) > timedelta(hours=gap_hours)
        if not new and last_located is not None:
            d = _km(last_located, a)
            new = d is not None and d > SPLIT_KM
        if new:
            events.append(Event())
        events[-1].items.append(a)
        if a.latitude is not None:
            last_located = a
    merged: list[Event] = []
    for e in events:
        prev = merged[-1] if merged else None
        if (
            prev and e.city and prev.city == e.city
            and (e.start.date() - prev.end.date()).days <= 1
            and (e.start - prev.end) <= timedelta(hours=36)
        ):
            prev.items.extend(e.items)
        else:
            merged.append(e)
    return merged


@dataclass
class Idea:
    key: str
    kind: str
    title: str
    reason: str
    params: dict
    preview_ids: list[str]
    score: float = 0.0  # ranking within one scan; not stored


KIND_WEIGHT = {"highlight": 1.0, "collage": 0.9, "popout": 0.6, "cinematic": 0.35}
EVENT_BOOST = {"birthday": 1.6, "festival": 1.4, "school": 1.3, "wedding": 1.4, "anniversary": 1.3,
               "trip": 1.2, "outing": 1.1, "get-together": 1.1, "event": 1.0}


def rank(ideas: list[Idea], limit: int) -> list[Idea]:
    """Best ideas first, with kinds mixed: no kind takes more than half the slots
    while other kinds still have ideas to offer."""
    pool = sorted(ideas, key=lambda i: -i.score)
    out: list[Idea] = []
    cap = max(1, -(-limit // 2))
    while pool and len(out) < limit:
        counts = {k: sum(o.kind == k for o in out) for k in KIND_WEIGHT}
        others_left = {i.kind for i in pool}
        pick = next((i for i in pool if counts[i.kind] < cap or others_left == {i.kind}), pool[0])
        out.append(pick)
        pool.remove(pick)
    return out


def _event_info(event: Event) -> dict:
    k = event.kind
    if not k:
        return {"title": event.title(), "kind": "event"}
    return {"title": k.title, "kind": k.kind, "person_id": k.person_id, "person_name": k.person_name}


def ideas_for(event: Event, settings: AppSettings) -> list[Idea]:
    good = [a for a in event.photos if not looks_like_screenshot(a.filename, a.mime, a.make)]
    out: list[Idea] = []
    title = event.title()
    why = f"{event.kind.reason} " if event.kind else ""
    ids = [a.id for a in good]
    # The pipelines re-fetch the event by its exact time span (stays small in the DB).
    span = {"asset_ids": [], "event_from": event.start.isoformat(), "event_to": event.end.isoformat(), "event": _event_info(event)}
    caption = title.split(" \u00b7 ")[0]
    if len(good) >= settings.suggestions.min_good_photos:
        step = max(1, len(ids) // 6)
        out.append(Idea(
            f"collage:{event.key}", "collage", title,
            f"{why}{len(good)} photos; none used in a collage before.",
            dict(span, caption=caption), ids[::step][:6],
        ))
    n_items = len(event.items)
    if n_items >= HIGHLIGHT_MIN_ITEMS or (event.videos and n_items >= 10):
        clips = f", {len(event.videos)} clip(s)" if event.videos else ""
        out.append(Idea(
            f"highlight:{event.key}", "highlight", title,
            f"{why}{n_items} photos and videos{clips} over {max(1, (event.end - event.start).days + 1)} day(s).",
            dict(span, caption=caption), ids[:: max(1, len(ids) // 6)][:6],
        ))
    groups = [a for a in good if 1 <= len(a.people) <= 6 and min(a.width or 0, a.height or 0) >= 1000]
    if groups and settings.popout.styles():
        best = max(groups, key=lambda a: (a.is_favorite, 2 <= len(a.people) <= 4, len(a.people)))
        out.append(Idea(
            f"popout:{best.id}", "popout", f"{title} \u00b7 pop-out",
            f"{why}A clear photo of {', '.join(best.people[:3])}; shown as colour splash, spotlight, portrait blur, frame and sticker.",
            {"asset_ids": [best.id], "caption": caption, "event": _event_info(event)}, [best.id],
        ))
    portraits = [a for a in good if a.people and len(a.people) <= 3 and min(a.width or 0, a.height or 0) >= 1000]
    if portraits:
        best = max(portraits, key=lambda a: (a.is_favorite, len(a.people)))
        out.append(Idea(
            f"cinematic:{best.id}", "cinematic", f"{best.filename} \u00b7 {title}",
            f"{why}A sharp photo of {', '.join(best.people[:3])}; people separate well from the background.",
            {"asset_ids": [best.id], "event": _event_info(event)}, [best.id],
        ))
    return out


def through_the_years(people: list[Person], today: date) -> list[Idea]:
    """In the two weeks before a named person's birthday: '<name> through the years'."""
    out = []
    for p in people:
        try:
            nxt = date(today.year, p.birth.month, p.birth.day)
        except ValueError:
            nxt = date(today.year, 3, 1)
        if nxt < today:
            nxt = date(today.year + 1, nxt.month, nxt.day)
        days = (nxt - today).days
        if 0 <= days <= YEARS_AHEAD_DAYS and today.year - p.birth.year >= 2:
            age = nxt.year - p.birth.year
            title = f"{p.name} through the years"
            out.append(Idea(
                f"years:{p.id}:{nxt.year}", "highlight", title,
                f"{p.name} turns {age} on {nxt:%d %b}; a film of {p.name} across the years.",
                {"asset_ids": [], "person_ids": [p.id], "caption": title, "subtitle": f"{p.birth.year}\u2013{nxt.year}",
                 "event": {"title": title, "kind": "birthday", "person_id": p.id, "person_name": p.name}},
                [], score=3.0,
            ))
    return out


class SuggestionStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, idea: Idea) -> bool:
        """False when this idea was suggested before."""
        cur = self.db.execute(
            "INSERT OR IGNORE INTO suggestions (key, kind, title, reason, params, preview_ids, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'new', ?)",
            (idea.key, idea.kind, idea.title, idea.reason, json.dumps(idea.params), json.dumps(idea.preview_ids), now_iso()),
        )
        return cur.rowcount == 1

    def seen(self, key: str) -> bool:
        return self.db.one("SELECT 1 FROM suggestions WHERE key = ?", (key,)) is not None

    def clear_new(self) -> int:
        """Forget ideas nobody acted on, so a rescan can offer them again (testing aid)."""
        return self.db.execute("DELETE FROM suggestions WHERE status = 'new'").rowcount

    def purge(self, older_than_days: int, now: datetime | None = None) -> int:
        """Drop handled (approved / dismissed) suggestions older than N days, keeping their keys'
        effect: a dismissed idea is remembered as a key-only row so it isn't offered again."""
        if older_than_days <= 0:
            return 0
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        return self.db.execute(
            "UPDATE suggestions SET params = '{}', preview_ids = '[]', reason = '' "
            "WHERE status != 'new' AND created_at < ? AND params != '{}'", (cutoff,),
        ).rowcount

    def list(self, status: str = "new") -> list[dict]:
        rows = self.db.query("SELECT * FROM suggestions WHERE status = ? ORDER BY id DESC", (status,))
        return [dict(r, params=json.loads(r["params"]), preview_ids=json.loads(r["preview_ids"])) for r in rows]

    def approve(self, sid: int, jobs: JobStore) -> int | None:
        row = self.db.one("SELECT * FROM suggestions WHERE id = ? AND status = 'new'", (sid,))
        if row is None:
            return None
        params = json.loads(row["params"])
        job = jobs.enqueue(row["kind"], row["title"], params, origin="suggestion")
        self.db.execute("UPDATE suggestions SET status = 'approved', job_id = ? WHERE id = ?", (job.id, sid))
        return job.id

    def dismiss(self, sid: int) -> bool:
        return self.db.execute("UPDATE suggestions SET status = 'dismissed' WHERE id = ? AND status = 'new'", (sid,)).rowcount == 1

    def last_scan(self) -> datetime | None:
        row = self.db.one("SELECT value FROM meta WHERE key = 'last_scan'")
        return datetime.fromisoformat(row["value"]) if row else None

    def set_last_scan(self, when: datetime) -> None:
        self.db.execute(
            "INSERT INTO meta (key, value) VALUES ('last_scan', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (when.isoformat(timespec="seconds"),),
        )


async def _ocr_kinds(gateway: ImmichGateway, event: Event) -> list[str]:
    if len(event.items) < OCR_MIN_ITEMS:
        return []
    found = []
    a, b = event.start - timedelta(seconds=1), event.end + timedelta(seconds=1)
    for text, kind, need in OCR_HINTS:
        if kind in found:
            continue
        try:
            if await gateway.ocr_hits(text, a, b) >= need:
                found.append(kind)
        except ImmichError:
            return found  # OCR off on this server: skip the signal
    return found


def merge_same_titles(events: list[Event]) -> list[Event]:
    """Neighbouring events that came out as the same occasion (two halves of a birthday party
    split by a long lunch, two days of one trip) become one."""
    out: list[Event] = []
    for e in events:
        prev = out[-1] if out else None
        if (
            prev and prev.kind and e.kind and prev.kind.kind != "event" and prev.kind.title == e.kind.title
            and (e.start.date() - prev.end.date()).days <= 1
        ):
            prev.items.extend(e.items)
        else:
            out.append(e)
    return out


def trips(events: list[Event], min_days: int = 2) -> list[Event]:
    """Runs of consecutive days away from home (outings / trips in different towns) as one
    trip, for a highlight of the whole journey. The day-by-day events stay as they are."""
    out: list[Event] = []
    run: list[Event] = []

    def close() -> None:
        if len(run) >= 2:
            days = (run[-1].end.date() - run[0].start.date()).days + 1
            if days >= min_days:
                t = Event(items=[a for e in run for a in e.items])
                places = []
                for e in run:
                    if e.city and e.city not in places:
                        places.append(e.city)
                a, b = t.start, t.end
                when = f"{a.day}\u2013{b.day} {b:%b %Y}" if a.month == b.month else f"{a.day} {a:%b} \u2013 {b.day} {b:%b %Y}"
                t.kind = EventKind("trip", f"Trip \u00b7 {when}", f"{days} days away from home: {', '.join(places[:5])}{' and more' if len(places) > 5 else ''}.", signals=["place"])
                out.append(t)
        run.clear()

    for e in events:
        away = e.kind is not None and e.kind.kind in ("outing", "trip", "wedding")
        if away and run and (e.start.date() - run[-1].end.date()).days <= 1:
            run.append(e)
        elif away:
            close()
            run.append(e)
        elif len(e.items) >= 10 or (run and (e.start.date() - run[-1].end.date()).days > 1):
            close()  # a real day at home ends the trip; a few stray photos in between don't
    close()
    return out


async def classify_events(gateway: ImmichGateway, events: list[Event], people: list[Person], assets: list[AssetInfo]) -> list[Event]:
    located = [x for x in assets if x.latitude is not None]
    home = home_location([(x.latitude, x.longitude) for x in located], [x.taken_at.date() for x in located])
    for e in events:
        e.kind = classify(
            days=[x.taken_at.date() for x in e.items],
            people_in_photos=[n for x in e.items for n in x.people],
            locations=[(x.latitude, x.longitude) for x in e.items if x.latitude is not None],
            place=e.city,
            birthdays=people,
            ocr_kinds=await _ocr_kinds(gateway, e),
            home=home,
        )
    return merge_same_titles(events)


async def load_people(gateway: ImmichGateway) -> list[Person]:
    try:
        return [Person(i, n, b) for i, n, b in await gateway.people_with_birthdays()]
    except ImmichError:
        return []


async def scan(gateway: ImmichGateway, store: SuggestionStore, jobs: JobStore, settings: AppSettings, now: datetime | None = None) -> dict:
    """One scan. Returns counts for the UI."""
    now = now or datetime.now(timezone.utc)
    since = store.last_scan() or now - timedelta(days=FIRST_SCAN_DAYS)
    assets = [a for a in await gateway.search(taken_after=since, taken_before=now, limit=5000) if a.usable]
    events = group_events(assets, settings.suggestions.event_gap_hours)
    people = await load_people(gateway)
    events = await classify_events(gateway, events, people, assets)
    span = max((now - since).total_seconds(), 1.0)
    candidates: list[Idea] = []
    for event in events:
        recency = max(0.0, 1 - (now - event.end).total_seconds() / span)  # 1 = today
        boost = EVENT_BOOST.get(event.kind.kind if event.kind else "event", 1.0)
        for idea in ideas_for(event, settings):
            if store.seen(idea.key):
                continue
            idea.score = KIND_WEIGHT[idea.kind] * math.log1p(len(event.items)) * (1 + 0.5 * recency) * boost
            candidates.append(idea)
    for trip in trips(events):
        idea = next((i for i in ideas_for(trip, settings) if i.kind == "highlight"), None)
        if idea and not store.seen(idea.key):
            recency = max(0.0, 1 - (now - trip.end).total_seconds() / span)
            idea.score = KIND_WEIGHT["highlight"] * math.log1p(len(trip.items)) * (1 + 0.5 * recency) * 1.3
            candidates.append(idea)
    candidates += [i for i in through_the_years(people, now.date()) if not store.seen(i.key)]
    added = 0
    for idea in rank(candidates, settings.suggestions.max_per_day):
        if store.add(idea):
            added += 1
    queued = 0
    if settings.suggestions.mode == "auto":
        for s in store.list("new"):
            if store.approve(s["id"], jobs):
                queued += 1
    store.set_last_scan(now)
    kinds = sorted({e.kind.kind for e in events if e.kind})
    log.info("Suggestion scan: %d items, %d events (%s), %d ideas, %d queued", len(assets), len(events), ", ".join(kinds), added, queued)
    return {"checked": len(assets), "events": len(events), "added": added, "queued": queued, "since": since,
            "event_kinds": [(e.title(), e.kind.kind if e.kind else "event") for e in events]}


def due(settings: AppSettings, last: datetime | None, now: datetime) -> bool:
    """True once a day, at or after the scan time (local server time)."""
    if settings.suggestions.mode == "off":
        return False
    hh, mm = map(int, settings.suggestions.scan_at.split(":"))
    local = now.astimezone()
    today_at = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if local < today_at:
        return False
    return last is None or last.astimezone() < today_at
