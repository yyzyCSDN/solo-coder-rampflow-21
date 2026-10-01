from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import Iterable, Mapping

from .timeutil import Window, utc


@dataclass(frozen=True)
class FlightRecoveryInput:
    flight_id: str
    scheduled_off: datetime
    passengers: int
    connections: int
    curfew: datetime | None
    minimum_turn_min: int
    current_ready: datetime
    cancellation_cost: int

    def __post_init__(self):
        object.__setattr__(self, "scheduled_off", utc(self.scheduled_off))
        object.__setattr__(self, "current_ready", utc(self.current_ready))
        if self.curfew is not None:
            object.__setattr__(self, "curfew", utc(self.curfew))


@dataclass(frozen=True)
class RecoveryOption:
    flight_id: str
    action: str
    departure: datetime | None
    stand: str | None
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryPlan:
    options: Mapping[str, RecoveryOption]
    total_score: int
    generated_at: datetime


class DisruptionOptimizer:
    def __init__(
        self,
        delay_cost_per_minute: int = 2,
        connection_cost: int = 80,
        passenger_cost: int = 1,
    ):
        self.delay_cost_per_minute = delay_cost_per_minute
        self.connection_cost = connection_cost
        self.passenger_cost = passenger_cost

    def options_for(
        self,
        flight: FlightRecoveryInput,
        candidate_stands: Iterable[str],
        stand_windows: Mapping[str, tuple[Window, ...]],
        delay_steps: Iterable[int] = (0, 10, 20, 30, 45, 60, 90),
    ) -> tuple[RecoveryOption, ...]:
        out = []
        for delay in delay_steps:
            departure = max(
                flight.scheduled_off + timedelta(minutes=delay),
                flight.current_ready,
            )
            if flight.curfew is not None and departure > flight.curfew:
                continue
            delay_min = max(
                0,
                int((departure - flight.scheduled_off).total_seconds() // 60),
            )
            for stand in candidate_stands:
                blocked = any(w.contains(departure) for w in stand_windows.get(stand, ()))
                if blocked:
                    continue
                score = (
                    delay_min * self.delay_cost_per_minute
                    + min(flight.connections, delay_min // 10) * self.connection_cost
                    + (flight.passengers * self.passenger_cost if delay_min > 60 else 0)
                )
                reasons = []
                if departure == flight.current_ready and flight.current_ready > flight.scheduled_off:
                    reasons.append("ready-time-limited")
                if delay_min:
                    reasons.append(f"delay:{delay_min}")
                out.append(
                    RecoveryOption(
                        flight.flight_id,
                        "operate",
                        departure,
                        stand,
                        score,
                        tuple(reasons),
                    )
                )
        out.append(
            RecoveryOption(
                flight.flight_id,
                "cancel",
                None,
                None,
                flight.cancellation_cost,
                ("cancellation",),
            )
        )
        return tuple(sorted(out, key=lambda o: (o.score, o.action, o.departure or datetime.max.replace(tzinfo=flight.scheduled_off.tzinfo))))

    def optimize(
        self,
        flights: Iterable[FlightRecoveryInput],
        candidate_stands: Mapping[str, tuple[str, ...]],
        stand_windows: Mapping[str, tuple[Window, ...]],
        generated_at: datetime,
        incompatible_stands: Mapping[str, frozenset[str]] | None = None,
    ) -> RecoveryPlan:
        flights = tuple(flights)
        if incompatible_stands is None:
            incompatible_stands = {}
        option_sets = [
            self.options_for(f, candidate_stands.get(f.flight_id, ()), stand_windows)
            for f in flights
        ]

        def incompatible(a: str, b: str) -> bool:
            if a == b:
                return True
            pair = incompatible_stands.get(a, frozenset())
            reverse = incompatible_stands.get(b, frozenset())
            # Adjacency isolation is a physical, symmetric restriction even if
            # the rule is declared on one side only.
            return b in pair or a in reverse

        def feasible(combo: tuple[RecoveryOption, ...]) -> bool:
            # Safety limits are hard constraints: they may never be traded away
            # for a cheaper score.  The occupancy table is shared across all
            # flights in the combination, rather than rebuilt per option.
            used: dict[str, list[datetime]] = {}
            for option in combo:
                if option.action == "cancel":
                    continue
                assert option.departure is not None and option.stand is not None
                for occupied_stand, departures in used.items():
                    if not incompatible(option.stand, occupied_stand):
                        continue
                    for existing in departures:
                        if abs((existing - option.departure).total_seconds()) < 15 * 60:
                            return False
                used.setdefault(option.stand, []).append(option.departure)
            return True

        def search(truncate: bool):
            sets = [opts[:8] for opts in option_sets] if truncate else option_sets
            best = None
            for combo in product(*sets):
                if not feasible(combo):
                    continue
                score = sum(o.score for o in combo)
                tie = tuple(
                    (o.flight_id, o.action, o.departure or generated_at, o.stand or "")
                    for o in combo
                )
                candidate = (score, tie, combo)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            return best

        best = search(truncate=True)
        if best is None:
            # A feasible combination could live outside the per-flight top-N.
            # Retry exhaustively before declaring the disruption unrecoverable.
            best = search(truncate=False)
        if best is None:
            raise RuntimeError("no disruption recovery plan")
        combo = best[2]
        return RecoveryPlan({o.flight_id: o for o in combo}, best[0], utc(generated_at))

