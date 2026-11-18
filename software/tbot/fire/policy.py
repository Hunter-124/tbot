"""The bar: how sure the solver must be before it spends the one shot of a press.

A fixed bar is wrong in both directions. P(hit) is capped by what cannot be known - where a hand-held gun will
point one trigger delay from now - and that cap moves from 0.9 on a smooth swing to 0.3 on a rough one or at long
range. A bar of 0.40 then waits for something a rough swing cannot give (no shot at all: a clay lost for certain),
while on a smooth swing it is cleared long before the aim is at its best. So the bar follows the engagement:

  reach    never ask for more than `reach` x the P(hit) a PERFECT aim would have right now (zero predicted
           miss, the present uncertainty): when the aim is that close to the best this swing allows, waiting
           buys nothing;
  regret   a chance this target already offered and lost (P rose to a top below the bar, then fell to less than
           half of it) is the measure of the next one: `regret` x that top is enough the second time;
  relax    the longer a target has been followed without a shot, the lower the bar, sliding from `bar` to
           `relaxed` between two track ages (the range-day rule was the same thing as a step);
  floor    and never below `floor`: under it the charge is more likely wasted than not, whatever the excuse.

The solver still fires only at a top of P(hit) (nothing in its look-ahead promises more); this decides whether
that top is worth the shot. `tbot vclay --policies` measures each part on recorded swings: `reach` does most of
the work, and makes the result insensitive to where the bar starts.
"""

from __future__ import annotations

LOST_FRACTION = 0.5  # a top counts as a lost chance once P has fallen to this share of it


class FirePolicy:
    __slots__ = ("base", "floor", "reach", "regret", "relax_from_s", "relax_to_s", "relaxed",
                 "_track", "_top", "_lost")

    def __init__(self, bar: float, floor: float, reach: float = 0.0, regret: float = 0.0,
                 relax_from_s: float = 0.0, relax_to_s: float = 0.0, relaxed: float = 0.0) -> None:
        self.base = float(bar)
        self.floor = min(float(floor), self.base) if floor > 0.0 else self.base
        self.reach = float(reach)
        self.regret = float(regret)
        self.relax_from_s = float(relax_from_s)
        self.relax_to_s = max(float(relax_to_s), self.relax_from_s)
        self.relaxed = float(relaxed) if 0.0 < relaxed < self.base else self.base
        self._track = None
        self._top = 0.0
        self._lost = 0.0

    @classmethod
    def from_config(cls, cfg) -> "FirePolicy":
        f = cfg.fire
        bar = float(f.min_hit_probability)
        settled = float(getattr(f, "min_hit_probability_settled", 0.0) or 0.0)
        if not bool(getattr(f, "adaptive", False)):
            # the fixed rule: `bar`, and the settled bar once the track is settled_ms old
            step = float(getattr(f, "settled_ms", 0.0)) * 1e-3
            return cls(bar, floor=settled if 0.0 < settled < bar else bar, relax_from_s=step, relax_to_s=step,
                       relaxed=settled)
        return cls(bar, floor=float(f.min_hit_probability_floor), reach=float(f.adaptive_reach),
                   regret=float(f.adaptive_regret), relax_from_s=float(f.adaptive_relax_from_ms) * 1e-3,
                   relax_to_s=float(f.adaptive_relax_to_ms) * 1e-3, relaxed=settled)

    def reset(self) -> None:
        self._track = None
        self._top = 0.0
        self._lost = 0.0

    def bar(self, track_id, now_ns: int, first_t_ns: int, p_hit: float, ceiling: float) -> float:
        """The bar for this frame's solution. `ceiling` is the P(hit) a perfect aim would have with the present
        uncertainty. Call it once per frame with a solution (it remembers the target's earlier chances)."""
        if track_id != self._track:
            self._track, self._top, self._lost = track_id, 0.0, 0.0
        bar = self.base
        # relax: the longer the target has been followed, the less is asked
        if self.relaxed < bar and first_t_ns:
            age = (int(now_ns) - int(first_t_ns)) * 1e-9
            if age >= self.relax_to_s:
                bar = self.relaxed
            elif age > self.relax_from_s:
                bar -= (bar - self.relaxed) * (age - self.relax_from_s) / (self.relax_to_s - self.relax_from_s)
        # reach: what a perfect aim would be worth with the present uncertainty
        if self.reach > 0.0 and ceiling == ceiling:
            bar = min(bar, self.reach * float(ceiling))
        # regret: the best chance this target has already let go
        if self.regret > 0.0 and self._lost > 0.0:
            bar = min(bar, self.regret * self._lost)
        bar = max(bar, self.floor)
        # remember this frame for the next ones
        p = float(p_hit)
        if p == p:  # not NaN
            if p > self._top:
                self._top = p
            elif p < LOST_FRACTION * self._top:
                if self._top > self._lost:
                    self._lost = self._top
                self._top = p
        return bar
