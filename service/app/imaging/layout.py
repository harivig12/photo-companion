"""Collage layout planner.

Photos keep their chronological order. Three families of layouts are tried and the one that
needs the least cropping wins:

- **rows**: photos split into 1-4 rows of 1-4 photos; every row is scaled so its photos fill
  the width at their natural aspect ("justified" rows). Rows are then stretched together to
  fill the canvas height; that stretch is the crop each photo pays.
- **columns**: the same, turned 90 degrees.
- **hero** (3+ photos): the best photo gets a large slot (left or top, 50-67 % of the canvas)
  and the rest fill the remaining area with rows or columns.

Cost per photo = |ln(photo aspect / slot aspect)|, i.e. how much has to be cropped away.
Small penalties discourage uneven rows and tiny slots. A hero layout gets a small bonus so it
wins close calls (it reads better at a glance, like Google's templates).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

MAX_PER_ROW = 4
MAX_ROWS = 4
HERO_FRACTIONS = (0.5, 0.6, 2 / 3)
HERO_BONUS = 0.06
UNEVEN_WEIGHT = 0.12
TINY_WEIGHT = 0.5


@dataclass(frozen=True)
class Slot:
    """A rectangle in canvas units: the canvas is 1 wide and 1/ratio tall."""

    x: float
    y: float
    w: float
    h: float

    @property
    def aspect(self) -> float:
        return self.w / self.h


@dataclass(frozen=True)
class Layout:
    name: str
    slots: tuple[Slot, ...]  # slots[i] holds photo i (input order)
    cost: float
    hero: int | None = None


def parse_ratio(text: str) -> float:
    w, h = text.split(":")
    return float(w) / float(h)


@lru_cache(maxsize=None)
def _compositions(n: int) -> tuple[tuple[int, ...], ...]:
    """All ways to split n ordered items into contiguous groups (sizes), within the limits."""
    out = []
    for k in range(1, min(n, MAX_ROWS) + 1):
        for cuts in combinations(range(1, n), k - 1):
            bounds = (0, *cuts, n)
            sizes = tuple(bounds[i + 1] - bounds[i] for i in range(k))
            if max(sizes) <= MAX_PER_ROW:
                out.append(sizes)
    return tuple(out)


def _rows(aspects: Sequence[float], width: float, height: float, x0: float = 0.0, y0: float = 0.0) -> tuple[float, list[Slot], str] | None:
    """Best justified-rows layout of `aspects` inside a width x height box."""
    n = len(aspects)
    if n == 0:
        return None
    best: tuple[float, list[Slot], str] | None = None
    area_floor = (width * height) / (3.0 * n)
    for sizes in _compositions(n):
        heights, idx = [], 0
        for size in sizes:
            heights.append(width / sum(aspects[idx : idx + size]))
            idx += size
        stretch = height / sum(heights)
        crop = abs(math.log(stretch))
        slots: list[Slot] = []
        y, idx = y0, 0
        tiny = 0.0
        for size, h in zip(sizes, heights):
            x = x0
            rh = h * stretch
            for a in aspects[idx : idx + size]:
                w = a * h
                slots.append(Slot(x, y, w, rh))
                if w * rh < area_floor:
                    tiny += 1
                x += w
            y += rh
            idx += size
        uneven = math.log(max(heights) / min(heights)) if len(heights) > 1 else 0.0
        cost = crop + UNEVEN_WEIGHT * uneven + TINY_WEIGHT * tiny / n
        if best is None or cost < best[0]:
            best = (cost, slots, "rows " + "-".join(map(str, sizes)))
    return best


def _columns(aspects: Sequence[float], width: float, height: float, x0: float = 0.0, y0: float = 0.0) -> tuple[float, list[Slot], str] | None:
    res = _rows([1 / a for a in aspects], height, width, y0, x0)
    if res is None:
        return None
    cost, slots, name = res
    return cost, [Slot(s.y, s.x, s.h, s.w) for s in slots], name.replace("rows", "columns")


def plan(aspects: Sequence[float], scores: Sequence[float], canvas_ratio: float) -> Layout:
    """Pick the layout with the least cropping for photos with these aspects (w/h)."""
    n = len(aspects)
    if n < 1:
        raise ValueError("need at least one photo")
    if len(scores) != n:
        raise ValueError("one score per photo")
    W, H = 1.0, 1.0 / canvas_ratio
    candidates: list[Layout] = []

    for family in (_rows, _columns):
        res = family(aspects, W, H)
        if res:
            cost, slots, name = res
            candidates.append(Layout(name, tuple(slots), cost))

    if n >= 3:
        hero = max(range(n), key=lambda i: scores[i])
        rest_idx = [i for i in range(n) if i != hero]
        rest = [aspects[i] for i in rest_idx]
        for frac in HERO_FRACTIONS:
            for side in ("left", "top"):
                if side == "left":
                    hero_slot = Slot(0, 0, W * frac, H)
                    sub = _columns(rest, W * (1 - frac), H, W * frac, 0) if len(rest) <= MAX_PER_ROW * MAX_ROWS else None
                    sub_alt = _rows(rest, W * (1 - frac), H, W * frac, 0)
                else:
                    hero_slot = Slot(0, 0, W, H * frac)
                    sub = _rows(rest, W, H * (1 - frac), 0, H * frac)
                    sub_alt = _columns(rest, W, H * (1 - frac), 0, H * frac)
                for s in (sub, sub_alt):
                    if s is None:
                        continue
                    sub_cost, sub_slots, sub_name = s
                    hero_cost = abs(math.log(aspects[hero] / hero_slot.aspect))
                    cost = (hero_cost + sub_cost * (n - 1)) / n - HERO_BONUS
                    slots: list[Slot] = [None] * n  # type: ignore[list-item]
                    slots[hero] = hero_slot
                    for i, sl in zip(rest_idx, sub_slots):
                        slots[i] = sl
                    candidates.append(Layout(f"hero {side} {round(frac * 100)}% + {sub_name}", tuple(slots), cost, hero))

    return min(candidates, key=lambda c: c.cost)
