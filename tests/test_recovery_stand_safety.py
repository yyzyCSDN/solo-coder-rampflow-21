import unittest
from datetime import datetime, timezone

from rampflow.disruption_optimizer import DisruptionOptimizer, FlightRecoveryInput
from rampflow.models import Decision
from rampflow.stand_orchestration import (
    StandConflict,
    StandDemand,
    StandOrchestrator,
    StandProfile,
)
from rampflow.timeutil import Window
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc


def D(hour, minute=0):
    return datetime(2026, 1, 1, hour, minute, tzinfo=UTC)


def flight_input(flight_id, off, ready=None, curfew=None, cancellation_cost=100000):
    return FlightRecoveryInput(
        flight_id=flight_id,
        scheduled_off=off,
        passengers=0,
        connections=0,
        curfew=curfew,
        minimum_turn_min=30,
        current_ready=ready if ready is not None else off,
        cancellation_cost=cancellation_cost,
    )


class StandIsolationRuleTests(unittest.TestCase):
    def test_reverse_direction_incompatibility_blocks_stand(self):
        # A1 declares A2 incompatible; A2 does not declare anything.
        # Placing on A2 must still respect A1's declaration.
        orch = StandOrchestrator([
            StandProfile("A1", "T1", 60, True, 1, incompatible_simultaneous=frozenset({"A2"})),
            StandProfile("A2", "T1", 60, True, 1),
        ])
        orch.commit(orch.plan(StandDemand("F1", Window(D(10), D(11)), 35, False)))
        with self.assertRaises(StandConflict):
            orch.plan(StandDemand("F2", Window(D(10, 30), D(11, 30)), 35, False))

    def test_forward_direction_incompatibility_still_blocks(self):
        orch = StandOrchestrator([
            StandProfile("A1", "T1", 60, True, 1, incompatible_simultaneous=frozenset({"A2"})),
            StandProfile("A2", "T1", 60, True, 1),
        ])
        orch.commit(orch.plan(StandDemand("F1", Window(D(10), D(11)), 35, False)))
        with self.assertRaises(StandConflict):
            orch.plan(StandDemand("F2", Window(D(10, 30), D(11, 30)), 35, False))

    def test_incompatible_pair_allowed_when_windows_do_not_overlap(self):
        orch = StandOrchestrator([
            StandProfile("A1", "T1", 60, True, 1, incompatible_simultaneous=frozenset({"A2"})),
            StandProfile("A2", "T1", 60, True, 1),
        ])
        orch.commit(orch.plan(StandDemand("F1", Window(D(10), D(11)), 35, False)))
        later = orch.plan(StandDemand("F2", Window(D(12), D(13)), 35, False))
        self.assertIn(later.stand_id, {"A1", "A2"})


class OptimizerFeasibilityTests(unittest.TestCase):
    def test_same_stand_departures_are_separated(self):
        optimizer = DisruptionOptimizer()
        flights = [
            flight_input("F1", D(10)),
            flight_input("F2", D(10, 5)),
        ]
        plan = optimizer.optimize(
            flights,
            {"F1": ("S1",), "F2": ("S1",)},
            {},
            D(9),
        )
        first = plan.options["F1"]
        second = plan.options["F2"]
        self.assertEqual(first.action, "operate")
        self.assertEqual(second.action, "operate")
        gap = abs((first.departure - second.departure).total_seconds()) / 60
        self.assertGreaterEqual(gap, 15)

    def test_incompatible_stands_not_traded_against_cost(self):
        optimizer = DisruptionOptimizer()
        flights = [
            flight_input("F1", D(10)),
            flight_input("F2", D(10)),
        ]
        # Declared in one direction only; the cheapest combo (both on time,
        # one on S1 and one on S2) is unsafe and must not be selected.
        plan = optimizer.optimize(
            flights,
            {"F1": ("S1", "S2"), "F2": ("S1", "S2")},
            {},
            D(9),
            incompatible_stands={"S1": frozenset({"S2"})},
        )
        first = plan.options["F1"]
        second = plan.options["F2"]
        self.assertEqual(first.action, "operate")
        self.assertEqual(second.action, "operate")
        gap = abs((first.departure - second.departure).total_seconds()) / 60
        self.assertGreaterEqual(gap, 15)
        # Cheapest feasible trade-off: delay one flight by 20 minutes (score 40)
        # rather than cancelling anything.
        self.assertEqual(plan.total_score, 40)

    def test_cancel_fallback_survives_candidate_cap(self):
        optimizer = DisruptionOptimizer()
        curfew = D(10, 25)
        flights = [flight_input(f"F{i}", D(10), curfew=curfew) for i in range(3)]
        stands = ("S1", "S2", "S3")
        plan = optimizer.optimize(
            flights,
            {f.flight_id: stands for f in flights},
            {},
            D(9),
            incompatible_stands={
                "S1": frozenset({"S2", "S3"}),
                "S2": frozenset({"S3"}),
            },
        )
        operating = [o for o in plan.options.values() if o.action == "operate"]
        cancelled = [o for o in plan.options.values() if o.action == "cancel"]
        # Three aircraft cannot all be separated by 15 minutes across mutually
        # isolated stands before the curfew, so at least one must cancel.
        self.assertGreaterEqual(len(cancelled), 1)
        for i, left in enumerate(operating):
            for right in operating[i + 1:]:
                gap = abs((left.departure - right.departure).total_seconds()) / 60
                self.assertGreaterEqual(gap, 15)


class RecoveryPlanPublicationTests(unittest.TestCase):
    def test_infeasible_plan_is_not_published(self):
        control = RampWorkflowControl()
        denied = Decision(False, "stand-incompatible", "stands A1/A2 mutually isolated")
        with self.assertRaises(WorkflowRejected):
            control.publish_recovery_stand_plan(
                "op-1", "F1", D(10), lambda: "committed", requirements=(denied,)
            )
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_failed_action_publishes_nothing(self):
        control = RampWorkflowControl()

        def boom():
            raise RuntimeError("stand commit failed")

        with self.assertRaises(RuntimeError):
            control.publish_recovery_stand_plan("op-2", "F1", D(10), boom)
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_successful_plan_publishes_once_and_replays(self):
        control = RampWorkflowControl()
        calls = []
        feasible = Decision(True, "ok", "stands feasible")
        first = control.publish_recovery_stand_plan(
            "op-3",
            "F1",
            D(10),
            lambda: calls.append("commit") or "plan",
            requirements=(feasible,),
        )
        second = control.publish_recovery_stand_plan(
            "op-3", "F1", D(10), lambda: calls.append("duplicate") or "plan"
        )
        self.assertIs(first, second)
        self.assertEqual(calls, ["commit"])
        self.assertEqual(control.stream.head, 1)
        self.assertEqual(first.event.topic, "recovery.stand.plan.published")


if __name__ == "__main__":
    unittest.main()
