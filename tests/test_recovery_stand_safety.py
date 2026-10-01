import unittest
from datetime import datetime, timezone

from rampflow.timeutil import Window
from rampflow.models import Decision
from rampflow.stand_orchestration import (
    StandProfile, StandDemand, StandOrchestrator, StandConflict,
)
from rampflow.disruption_optimizer import (
    DisruptionOptimizer, FlightRecoveryInput,
)
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected


UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


class AdjacencyDirectionTests(unittest.TestCase):
    def _profiles(self):
        # The isolation rule is declared on one side only: A says it cannot
        # coexist with B.  Physically the restriction is symmetric.
        return [
            StandProfile('A', 'T', 60, True, 1,
                         incompatible_simultaneous=frozenset({'B'})),
            StandProfile('B', 'T', 60, True, 1,
                         incompatible_simultaneous=frozenset()),
        ]

    def test_rule_declared_on_a_blocks_when_b_occupied_first(self):
        o = StandOrchestrator(self._profiles())
        first = o.plan(StandDemand('F1', Window(D(10), D(11)), 35, False))
        self.assertEqual(first.stand_id, 'A')
        o.commit(first)
        candidates = [
            a.stand_id
            for a in o.candidates(StandDemand('F2', Window(D(10, 30), D(11, 30)), 35, False))
        ]
        self.assertNotIn('B', candidates)

    def test_rule_declared_on_a_blocks_when_a_occupied_first(self):
        o = StandOrchestrator(self._profiles())
        first = o.plan(StandDemand('F1', Window(D(10), D(11)), 35, False))
        # Park on B directly (the side that declares no rule).
        o.commit(first.__class__(
            'F1', 'B', first.window, first.protected_window,
            first.generation, first.score,
        ))
        with self.assertRaises(StandConflict):
            o.plan(StandDemand('F2', Window(D(10, 30), D(11, 30)), 35, False))

    def test_non_overlapping_windows_still_allowed(self):
        o = StandOrchestrator(self._profiles())
        o.commit(o.plan(StandDemand('F1', Window(D(10), D(11)), 35, False)))
        second = o.candidates(StandDemand('F2', Window(D(13), D(14)), 35, False))
        self.assertEqual({a.stand_id for a in second}, {'A', 'B'})


class CombinationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.opt = DisruptionOptimizer()
        self.f1 = FlightRecoveryInput('F1', D(12), 100, 0, None, 30, D(11), 5000)
        self.f2 = FlightRecoveryInput('F2', D(12), 100, 0, None, 30, D(11), 5000)

    def test_two_flights_share_single_stand_without_double_booking(self):
        plan = self.opt.optimize(
            [self.f1, self.f2],
            {'F1': ('S1',), 'F2': ('S1',)},
            {}, D(9),
        )
        operating = [o for o in plan.options.values() if o.action == 'operate']
        self.assertEqual(len(operating), 2)
        gap = abs((operating[0].departure - operating[1].departure).total_seconds())
        self.assertGreaterEqual(gap, 15 * 60)

    def test_curfew_forces_cancellation_instead_of_overlap(self):
        f1 = FlightRecoveryInput('F1', D(12), 100, 0, D(12, 5), 30, D(11), 5000)
        f2 = FlightRecoveryInput('F2', D(12), 100, 0, D(12, 5), 30, D(11), 5000)
        plan = self.opt.optimize(
            [f1, f2], {'F1': ('S1',), 'F2': ('S1',)}, {}, D(9),
        )
        actions = sorted(o.action for o in plan.options.values())
        self.assertEqual(actions, ['cancel', 'operate'])

    def test_adjacent_isolating_stands_enforced_as_hard_constraint(self):
        plan = self.opt.optimize(
            [self.f1, self.f2],
            {'F1': ('A',), 'F2': ('B',)},
            {}, D(9),
            incompatible_stands={'A': frozenset({'B'})},
        )
        operating = [o for o in plan.options.values() if o.action == 'operate']
        self.assertEqual(len(operating), 2)
        gap = abs((operating[0].departure - operating[1].departure).total_seconds())
        self.assertGreaterEqual(gap, 15 * 60)

    def test_adjacency_rule_honoured_in_reverse_declaration(self):
        plan = self.opt.optimize(
            [self.f1, self.f2],
            {'F1': ('A',), 'F2': ('B',)},
            {}, D(9),
            incompatible_stands={'B': frozenset({'A'})},
        )
        operating = [o for o in plan.options.values() if o.action == 'operate']
        gap = abs((operating[0].departure - operating[1].departure).total_seconds())
        self.assertGreaterEqual(gap, 15 * 60)

    def test_distinct_stands_keep_cheapest_on_time_plan(self):
        plan = self.opt.optimize(
            [self.f1, self.f2],
            {'F1': ('A',), 'F2': ('B',)},
            {}, D(9),
        )
        for o in plan.options.values():
            self.assertEqual(o.action, 'operate')
            self.assertEqual(o.departure, D(12))
        self.assertEqual(plan.total_score, 0)


class RecoveryPublishGateTests(unittest.TestCase):
    def setUp(self):
        self.control = RampWorkflowControl()
        self.at = D(10)

    def test_rejected_safety_requirement_blocks_publish(self):
        calls = []
        denied = Decision(False, "adjacency", "mutually isolating stands")
        with self.assertRaises(WorkflowRejected):
            self.control.publish_recovery_stand_plan(
                "op-1", "PLAN-1", self.at,
                lambda: calls.append("committed"),
                requirements=(denied,),
            )
        self.assertEqual(calls, [])
        self.assertEqual(self.control.stream.head, 0)
        self.assertEqual(self.control.receipts, {})

    def test_domain_failure_leaves_no_published_event(self):
        def reject():
            raise RuntimeError("stand plan not executable")
        with self.assertRaises(RuntimeError):
            self.control.publish_recovery_stand_plan(
                "op-2", "PLAN-1", self.at, reject,
            )
        self.assertEqual(self.control.stream.head, 0)
        self.assertEqual(self.control.receipts, {})

    def test_success_publishes_once_and_is_idempotent(self):
        calls = []
        def commit():
            calls.append("committed")
            return "plan-v1"
        first = self.control.publish_recovery_stand_plan(
            "op-3", "PLAN-1", self.at, commit,
            requirements=(Decision(True, "ok", "feasible"),),
        )
        second = self.control.publish_recovery_stand_plan(
            "op-3", "PLAN-1", self.at, commit,
        )
        self.assertIs(first, second)
        self.assertEqual(calls, ["committed"])
        self.assertEqual(self.control.stream.head, 1)
        self.assertEqual(first.event.topic, "recovery.stand.plan.published")


if __name__ == "__main__":
    unittest.main()
