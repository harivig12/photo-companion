"""What kind of event a group of photos is: birthday, festival, school function, wedding,
get-together, outing or trip. All signals come from Immich, nothing leaves the network.

Signals, strongest first:
1. A named person's **birthday** (Immich stores birth dates) falls on the event's days.
2. **Festival dates**: fixed days, plus a table for festivals whose date moves (Deepavali,
   Vinayagar Chathurthi, Pongal). Matched within +/- 1 day: regional calendars differ by a day
   and evening photos cross midnight.
3. **Text in the photos** (Immich OCR, exact match): "happy birthday", "annual day", "wedding" ...
4. **Place**: far from home (the library's most common location) = outing (one day) or trip
   (several days); at home with 3+ named people = get-together.

Immich's smart (CLIP) search is not used: in v3.3 it returns results without a score, so it
always "finds" a cake. Festival tables: timeanddate.com (Naraka Chaturdasi 2012-2027, Ganesh
Chaturthi 2021-2031, Pongal 2021-2031), checked 24 Nov 2026; Vinayagar 2015-2020 and Deepavali
2028 from published calendars. Extend the tables yearly.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

# Tamil Deepavali is Naraka Chaturdasi (a day before North India's Lakshmi Puja in some years).
DEEPAVALI = {
    2012: (11, 12), 2013: (11, 2), 2014: (10, 22), 2015: (11, 10), 2016: (10, 29), 2017: (10, 18),
    2018: (11, 6), 2019: (10, 27), 2020: (11, 14), 2021: (11, 4), 2022: (10, 24), 2023: (11, 12),
    2024: (10, 31), 2025: (10, 20), 2026: (11, 8), 2027: (10, 28), 2028: (10, 17),
}
VINAYAGAR_CHATHURTHI = {
    2015: (9, 17), 2016: (9, 5), 2017: (8, 25), 2018: (9, 13), 2019: (9, 2), 2020: (8, 22),
    2021: (9, 10), 2022: (8, 31), 2023: (9, 19), 2024: (9, 7), 2025: (8, 27), 2026: (9, 14),
    2027: (9, 4), 2028: (8, 23), 2029: (9, 11), 2030: (9, 1), 2031: (9, 20),
}
PONGAL = {
    2021: (1, 14), 2022: (1, 14), 2023: (1, 15), 2024: (1, 15), 2025: (1, 14), 2026: (1, 14),
    2027: (1, 15), 2028: (1, 15), 2029: (1, 14), 2030: (1, 14), 2031: (1, 15),
}
FIXED = {
    "New Year": (1, 1), "Republic Day": (1, 26), "Independence Day": (8, 15), "Christmas": (12, 25),
}
MOVING = {"Deepavali": DEEPAVALI, "Vinayagar Chathurthi": VINAYAGAR_CHATHURTHI, "Pongal": PONGAL}

# OCR text -> event kind, and how many photos must show it. Phrases are strong (one photo is
# enough); single common words can be a sign in the background (a school bus, a "reception"
# desk), so they need at least two photos.
OCR_HINTS = (
    ("happy birthday", "birthday", 1), ("birthday", "birthday", 2),
    ("annual day", "school", 1), ("sports day", "school", 1), ("graduation", "school", 1), ("school", "school", 2),
    ("wedding", "wedding", 1), ("marriage", "wedding", 2), ("reception", "wedding", 2), ("anniversary", "anniversary", 2),
    ("happy diwali", "festival", 1), ("deepavali", "festival", 2), ("pongal", "festival", 2),
)
OUTING_KM = 40.0
KIND_LABEL = {
    "birthday": "Birthday", "festival": "Festival", "school": "School function", "wedding": "Wedding",
    "anniversary": "Anniversary", "get-together": "Get-together", "outing": "Outing", "trip": "Trip", "event": "Event",
}


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    birth: date


@dataclass
class EventKind:
    kind: str  # one of KIND_LABEL
    title: str  # album name / caption, e.g. "Atharv's 9th birthday", "Deepavali 2026", "Yercaud trip"
    reason: str
    person_id: str | None = None
    person_name: str | None = None
    signals: list[str] = field(default_factory=list)


def festival_on(day: date) -> str | None:
    """The festival on `day`: fixed days exactly, moving ones within +/- 1 day."""
    for name, (m, d) in FIXED.items():
        if date(day.year, m, d) == day:
            return name
    for name, table in MOVING.items():
        md = table.get(day.year)
        if md and abs((date(day.year, *md) - day).days) <= 1:
            return name
    return None


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def home_location(points: list[tuple[float, float]], days: list[date] | None = None) -> tuple[float, float] | None:
    """Home: the ~5 km grid cell photographed on the most *different days*.

    Counting photos would make a 500-photo trip "home"; counting days doesn't, because a trip
    lasts a few days while home shows up week after week. Without dates, photo counts are used.
    """
    if len(points) < 20:
        return None
    cell = lambda p: (round(p[0] / 0.05), round(p[1] / 0.05))  # noqa: E731
    if days and len(days) == len(points):
        per_cell: dict[tuple[int, int], set[date]] = {}
        for p, d in zip(points, days):
            per_cell.setdefault(cell(p), set()).add(d)
        best = max(per_cell, key=lambda c: len(per_cell[c]))
        if len(per_cell[best]) < 3:
            return None
    else:
        counts = Counter(cell(p) for p in points)
        best, n = counts.most_common(1)[0]
        if n < len(points) * 0.15:
            return None
    inside = [p for p in points if cell(p) == best]
    return sum(p[0] for p in inside) / len(inside), sum(p[1] for p in inside) / len(inside)


def classify(
    days: list[date],
    people_in_photos: list[str],
    locations: list[tuple[float, float]],
    place: str | None,
    birthdays: list[Person],
    ocr_kinds: list[str],
    home: tuple[float, float] | None,
) -> EventKind:
    first, last = min(days), max(days)
    span = {first + timedelta(days=i) for i in range((last - first).days + 1)}
    named = Counter(people_in_photos)
    place_s = f" in {place}" if place else ""

    # 1. Birthday of someone who is in the photos.
    for p in sorted(birthdays, key=lambda x: -named.get(x.name, 0)):
        if named.get(p.name, 0) == 0:
            continue
        for d in span | {first - timedelta(days=1), last + timedelta(days=1)}:
            try:
                bd = date(d.year, p.birth.month, p.birth.day)
            except ValueError:  # 29 Feb
                bd = date(d.year, 3, 1)
            if bd == d:
                age = d.year - p.birth.year
                title = f"{p.name}'s {ordinal(age)} birthday" if 0 < age < 120 else f"{p.name}'s birthday"
                return EventKind("birthday", title, f"{p.name}'s birthday ({p.birth:%d %b}) and {named[p.name]} photos of {p.name}.", p.id, p.name, ["birth date"])

    away = None
    if home and locations:
        away = sum(km(home, p) > OUTING_KM for p in locations) / len(locations) >= 0.5

    # 2. Festival: celebrated where you live, with most photos near the day. Far from home it's
    #    an outing or trip that happens to fall on a holiday.
    for d in sorted(span):
        f = festival_on(d)
        if not f or away:
            continue
        near = sum(abs((x - d).days) <= 1 for x in days) / len(days)
        if near >= 0.5:
            return EventKind("festival", f"{f} {d.year}", f"Taken on {f} ({d:%d %b %Y}).", signals=["festival date"])

    # 3. Text in the photos.
    if ocr_kinds:
        k = Counter(ocr_kinds).most_common(1)[0][0]
        if k == "birthday":
            return EventKind("birthday", f"Birthday celebration{place_s} \u00b7 {first:%b %Y}", "A 'happy birthday' sign is visible in the photos.", signals=["ocr"])
        if k == "festival":
            return EventKind("festival", f"Festival{place_s} \u00b7 {first:%b %Y}", "Festival greetings are visible in the photos.", signals=["ocr"])
        title = f"{KIND_LABEL[k]}{place_s} \u00b7 {first:%d %b %Y}"
        return EventKind(k, title, f"'{KIND_LABEL[k].lower()}' text is visible in the photos.", signals=["ocr"])

    # 4. Place: away from home = outing / trip; at home with several people = get-together.
    if away:
        if len(span) >= 2:
            return EventKind("trip", f"{place or 'Trip'} trip \u00b7 {first:%b %Y}" if place else f"Trip \u00b7 {first:%b %Y}", f"{len(span)} days, more than {OUTING_KM:.0f} km from home.", signals=["place"])
        return EventKind("outing", f"{place or 'Day'} outing \u00b7 {first:%d %b %Y}", f"A day more than {OUTING_KM:.0f} km from home.", signals=["place"])
    if len(named) >= 3:
        return EventKind("get-together", f"Get-together{place_s} \u00b7 {first:%d %b %Y}", f"{len(named)} people in the photos.", signals=["people"])
    when = f"{first:%d %b %Y}" if first == last else f"{first.day}\u2013{last.day} {last:%b %Y}" if first.month == last.month else f"{first:%d %b} \u2013 {last:%d %b %Y}"
    return EventKind("event", f"{place} \u00b7 {when}" if place else when, "Photos taken close together.")
