from datetime import date, datetime, timedelta, timezone

from app.events import Person, classify, festival_on, home_location, ordinal
from app.immich import AssetInfo
from app.suggestions import group_events, through_the_years

HOME = (11.66, 78.15)
YERCAUD = (11.78, 78.21)  # ~15 km: still "home" for the 40 km rule
GOA = (15.30, 74.12)


def cls(days, people=(), locs=(), place=None, birthdays=(), ocr=(), home=HOME):
    return classify(list(days), list(people), list(locs), place, list(birthdays), list(ocr), home)


def test_birthday_of_someone_in_the_photos_wins():
    atharv = Person("p1", "Atharv", date(2017, 8, 15))
    k = cls([date(2026, 8, 15)], people=["Atharv"] * 5 + ["Asha"], birthdays=[atharv])
    assert k.kind == "birthday" and k.title == "Atharv's 9th birthday" and k.person_id == "p1"
    # A day late (party on the weekend after) still counts; absent birthday person doesn't.
    assert cls([date(2026, 8, 16)], people=["Atharv"], birthdays=[atharv]).kind == "birthday"
    assert cls([date(2026, 8, 15)], people=["Asha"], birthdays=[atharv]).kind == "festival"  # Independence Day


def test_festivals_fixed_and_moving():
    assert festival_on(date(2026, 11, 8)) == "Deepavali"
    assert festival_on(date(2026, 11, 7)) == "Deepavali"  # Tamil calendars a day earlier
    assert festival_on(date(2026, 9, 14)) == "Vinayagar Chathurthi"
    assert festival_on(date(2027, 1, 15)) == "Pongal"
    assert festival_on(date(2026, 12, 25)) == "Christmas"
    assert festival_on(date(2026, 8, 16)) is None  # fixed days match exactly
    assert festival_on(date(2026, 10, 5)) is None
    k = cls([date(2026, 11, 8)])
    assert k.kind == "festival" and k.title == "Deepavali 2026"


def test_ocr_text_marks_school_and_wedding():
    k = cls([date(2026, 2, 10)], ocr=["school"], place="Salem")
    assert k.kind == "school" and k.title.startswith("School function in Salem")
    assert cls([date(2026, 2, 10)], ocr=["wedding"]).kind == "wedding"


def test_place_outing_trip_and_get_together():
    one_day = cls([date(2026, 10, 5)], locs=[GOA] * 10, place="Killai")
    assert one_day.kind == "outing" and one_day.title == "Killai outing \u00b7 05 Oct 2026"
    trip = cls([date(2026, 9, 13), date(2026, 9, 15)], locs=[GOA] * 10, place="Yercaud")
    assert trip.kind == "trip" and trip.title.startswith("Yercaud trip")
    home = cls([date(2026, 6, 1)], people=["A", "B", "C"], locs=[YERCAUD] * 4)
    assert home.kind == "get-together"
    plain = cls([date(2026, 6, 1)], people=["A"], locs=[HOME])
    assert plain.kind == "event"


def test_home_is_the_most_common_place():
    pts = [HOME] * 30 + [GOA] * 10
    h = home_location(pts)
    assert abs(h[0] - HOME[0]) < 0.05 and abs(h[1] - HOME[1]) < 0.05
    assert home_location([HOME] * 5) is None  # too few to tell


def test_ordinals():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st"]


def _a(i, when, city):
    return AssetInfo(id=f"id{i}", type="IMAGE", filename=f"f{i}", mime="image/jpeg", width=4, height=3, taken_at=when,
                     is_favorite=False, visibility="timeline", is_trashed=False, city=city)


def test_consecutive_days_in_one_city_are_one_trip():
    t0 = datetime(2026, 9, 13, 18, 0, tzinfo=timezone.utc)
    days = [t0 + timedelta(days=d, hours=h) for d in range(3) for h in (0, 1, 2)]
    events = group_events([_a(i, t, "Yercaud") for i, t in enumerate(days)], 3)
    assert len(events) == 1 and len(events[0].items) == 9
    # A different city the next day stays separate.
    other = [_a(100, t0 + timedelta(days=3), "Salem")]
    assert len(group_events([_a(i, t, "Yercaud") for i, t in enumerate(days)] + other, 3)) == 2


def test_through_the_years_in_the_fortnight_before_a_birthday():
    p = Person("p1", "Atharv", date(2017, 8, 15))
    ideas = through_the_years([p], date(2026, 8, 5))
    assert len(ideas) == 1 and ideas[0].kind == "highlight" and ideas[0].params["person_ids"] == ["p1"]
    assert ideas[0].params["event"]["kind"] == "birthday" and ideas[0].key == "years:p1:2026"
    assert through_the_years([p], date(2026, 7, 1)) == []  # too early
    assert through_the_years([Person("b", "Baby", date(2026, 1, 1))], date(2026, 12, 25)) == []  # < 2 years


def test_a_trip_over_a_festival_is_still_a_trip_but_home_festival_is_a_festival():
    days = [date(2026, 9, 13)] * 5 + [date(2026, 9, 14)] * 3 + [date(2026, 9, 15)] * 5
    assert cls(days, locs=[GOA] * 10, place="Yercaud").kind == "trip"
    assert cls([date(2026, 9, 13)] * 8, locs=[GOA] * 8, place="Yercaud").kind == "outing"  # one day away
    # At home, mostly on the day: festival. At home but the day is a small part: not.
    assert cls([date(2026, 9, 14)] * 6 + [date(2026, 9, 20)], locs=[HOME] * 7).kind == "festival"
    assert cls([date(2026, 9, 14)] + [date(2026, 9, 20)] * 6, people=["A"], locs=[HOME] * 7).kind != "festival"


def test_home_counts_days_not_photos():
    from datetime import date as d

    pts = [GOA] * 500 + [HOME] * 40
    days = [d(2026, 9, 14)] * 500 + [d(2026, 1, 1) + timedelta(days=i * 7) for i in range(40)]
    h = home_location(pts, days)
    assert abs(h[0] - HOME[0]) < 0.05  # a 500-photo trip is not home


def test_trips_join_consecutive_away_days_across_towns():
    from app.events import EventKind
    from app.suggestions import Event, trips

    t0 = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)

    def ev(day, kind, city, n=12):
        e = Event(items=[_a(day * 100 + i, t0 + timedelta(days=day, minutes=i), city) for i in range(n)])
        e.kind = EventKind(kind, f"{city} {kind}", "")
        return e

    events = [ev(0, "outing", "Trichy"), ev(1, "event", "x", n=3), ev(1, "outing", "Vallam"), ev(2, "trip", "Valangaiman"),
              ev(3, "outing", "Killai"), ev(6, "get-together", "Home")]
    found = trips(events)
    assert len(found) == 1 and len(found[0].items) == 4 * 12  # stray photos at home don't end it, nor join it
    assert found[0].kind.title == "Trip \u00b7 2\u20135 Oct 2026" and "Trichy" in found[0].kind.reason
