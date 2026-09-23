"""Regression tests for the Python bindings.

Runs under plain ``python -m unittest`` -- no pytest, no numpy, nothing to
install.  The point of these tests is the *binding*, not the solver: that
arguments survive the ctypes boundary with the right widths, that results come
back with the right lengths, and that errors surface as Python exceptions
rather than as silently wrong numbers.
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import igaos
from igaos import INF, IgaosError, Model, Status


class TestLinear(unittest.TestCase):
    def test_textbook_lp(self):
        m = Model(sense="maximize")
        x = m.add_variable(0, INF, cost=3.0, name="x")
        y = m.add_variable(0, INF, cost=5.0, name="y")
        m.add_constraint({x: 1.0}, upper=4.0)
        m.add_constraint({y: 2.0}, upper=12.0)
        m.add_constraint({x: 3.0, y: 2.0}, upper=18.0)
        r = m.solve(verbosity=0)
        self.assertTrue(r.optimal)
        self.assertAlmostEqual(r.objective, 36.0, places=7)
        self.assertAlmostEqual(r.x[0], 2.0, places=7)
        self.assertAlmostEqual(r.x[1], 6.0, places=7)
        self.assertEqual(len(r.row_duals), 3)
        self.assertEqual(len(r.reduced_costs), 2)
        m.close()

    def test_minimize_is_the_default(self):
        m = Model()
        self.assertEqual(m.sense, "minimize")
        x = m.add_variable(1.0, 10.0, cost=2.0)
        r = m.solve(verbosity=0)
        self.assertAlmostEqual(r.objective, 2.0, places=9)
        m.close()

    def test_infeasible_is_reported_not_guessed(self):
        m = Model()
        x = m.add_variable(0, 1)
        m.add_constraint({x: 1.0}, lower=2.0)
        r = m.solve(verbosity=0)
        self.assertEqual(r.status, Status.INFEASIBLE)
        self.assertFalse(r.feasible)
        m.close()

    def test_objective_offset(self):
        m = Model()
        x = m.add_variable(2.0, 2.0, cost=1.0)
        m.objective_offset = 10.0
        r = m.solve(verbosity=0)
        self.assertAlmostEqual(r.objective, 12.0, places=9)
        m.close()


class TestMixedInteger(unittest.TestCase):
    def test_knapsack(self):
        weights = [12, 7, 11, 8, 9]
        values = [24, 13, 23, 15, 16]
        capacity = 26
        m = Model(sense="maximize")
        cols = [m.add_binary(cost=v, name="item%d" % i) for i, v in enumerate(values)]
        m.add_constraint({c: w for c, w in zip(cols, weights)}, upper=capacity)
        r = m.solve(verbosity=0)
        self.assertTrue(r.optimal)
        # brute force the same instance
        best = 0
        for mask in range(1 << len(weights)):
            w = sum(weights[i] for i in range(len(weights)) if mask >> i & 1)
            if w <= capacity:
                best = max(best, sum(values[i] for i in range(len(values)) if mask >> i & 1))
        self.assertAlmostEqual(r.objective, best, places=7)
        for v in r.x:
            self.assertLess(abs(v - round(v)), 1e-6)
        m.close()

    def test_cut_statistics_are_exposed(self):
        m = Model(sense="maximize")
        cols = [m.add_integer(0, 5, cost=(i % 3) + 1) for i in range(8)]
        m.add_constraint({c: (i % 4) + 1 for i, c in enumerate(cols)}, upper=17.0)
        m.add_constraint({c: (i % 5) + 1 for i, c in enumerate(cols)}, upper=21.0)
        r = m.solve(verbosity=0)
        self.assertTrue(r.optimal)
        self.assertGreaterEqual(r.cuts_applied, 0)
        self.assertGreaterEqual(r.cut_rounds, 0)
        m.close()


class TestQuadratic(unittest.TestCase):
    def test_convex_qp(self):
        # min (x-3)^2 + (y-2)^2  s.t. x + y <= 4
        m = Model()
        x = m.add_variable(0, INF, cost=-6.0)
        y = m.add_variable(0, INF, cost=-4.0)
        m.objective_offset = 13.0
        m.set_quadratic(x, x, 2.0)
        m.set_quadratic(y, y, 2.0)
        m.add_constraint({x: 1.0, y: 1.0}, upper=4.0)
        r = m.solve(verbosity=0)
        self.assertTrue(r.optimal)
        self.assertAlmostEqual(r.x[0], 2.5, places=5)
        self.assertAlmostEqual(r.x[1], 1.5, places=5)
        self.assertAlmostEqual(r.objective, 0.5, places=5)
        self.assertIn("interior point", r.algorithm)
        m.close()

    def test_upper_triangle_entry_is_normalized(self):
        m = Model()
        m.add_variable(0, 1)
        m.add_variable(0, 1)
        m.set_quadratic(0, 1, 0.5)     # swapped internally, must not raise
        m.close()


class TestBulkLoad(unittest.TestCase):
    def test_csc_load_matches_incremental(self):
        # max 3x + 5y with the same three rows, loaded as a matrix
        m = Model(sense="maximize")
        m.load(num_rows=3, num_cols=2,
               colptr=[0, 2, 4],
               rowidx=[0, 2, 1, 2],
               values=[1.0, 3.0, 2.0, 2.0],
               obj=[3.0, 5.0],
               col_lower=[0.0, 0.0], col_upper=[INF, INF],
               row_lower=[-INF, -INF, -INF],
               row_upper=[4.0, 12.0, 18.0])
        r = m.solve(verbosity=0)
        self.assertAlmostEqual(r.objective, 36.0, places=7)
        m.close()

    def test_load_validates_shapes(self):
        m = Model()
        with self.assertRaises(ValueError):
            m.load(1, 2, colptr=[0, 1], rowidx=[0], values=[1.0])
        with self.assertRaises(ValueError):
            m.load(1, 2, colptr=[0, 1, 1], rowidx=[0, 0], values=[1.0])
        m.close()


class TestFileInterchange(unittest.TestCase):
    def test_mps_round_trip(self):
        m = Model(sense="maximize")
        x = m.add_variable(0, INF, cost=3.0, name="x")
        y = m.add_variable(0, INF, cost=5.0, name="y")
        m.add_constraint({x: 1.0}, upper=4.0, name="c0")
        m.add_constraint({y: 2.0}, upper=12.0, name="c1")
        m.add_constraint({x: 3.0, y: 2.0}, upper=18.0, name="c2")
        first = m.solve(verbosity=0).objective

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.mps")
            m.write_mps(path)
            self.assertTrue(os.path.exists(path))
            again = Model.from_mps(path)
            second = again.solve(verbosity=0).objective
            again.close()
        self.assertAlmostEqual(first, second, places=7)
        m.close()

    def test_missing_file_raises(self):
        m = Model()
        with self.assertRaises(IgaosError):
            m.read_mps("/definitely/not/here.mps")
        m.close()


class TestParametersAndErrors(unittest.TestCase):
    def test_round_trip_parameters(self):
        m = Model()
        m.set_param("time_limit", 12.5)
        self.assertAlmostEqual(m.get_param("time_limit"), 12.5, places=9)
        m.set_param("cuts", False)
        self.assertEqual(m.get_param("cuts"), 0)
        m.set_param("cut_rounds_root", 3)
        self.assertEqual(m.get_param("cut_rounds_root"), 3)
        m.close()

    def test_unknown_parameter_raises(self):
        m = Model()
        with self.assertRaises(IgaosError):
            m.set_param("there_is_no_such_parameter", 1)
        with self.assertRaises(IgaosError):
            m.get_param("there_is_no_such_parameter")
        m.close()

    def test_results_before_solve_raise(self):
        m = Model()
        m.add_variable(0, 1)
        with self.assertRaises(IgaosError):
            m.result()
        m.close()

    def test_bad_sense_raises(self):
        with self.assertRaises(ValueError):
            Model(sense="sideways")

    def test_bad_vartype_raises(self):
        m = Model()
        with self.assertRaises(ValueError):
            m.add_variable(0, 1, vtype="spicy")
        m.close()

    def test_context_manager_closes(self):
        with Model() as m:
            m.add_variable(0, 1, cost=1.0)
            m.solve(verbosity=0)
        self.assertIsNone(m._handle)

    def test_version_is_reported(self):
        self.assertTrue(igaos.version())


class TestAlgorithms(unittest.TestCase):
    """Every continuous path must agree on the same model."""

    def _model(self):
        m = Model(sense="maximize")
        x = m.add_variable(0, INF, cost=3.0)
        y = m.add_variable(0, INF, cost=5.0)
        m.add_constraint({x: 1.0}, upper=4.0)
        m.add_constraint({y: 2.0}, upper=12.0)
        m.add_constraint({x: 3.0, y: 2.0}, upper=18.0)
        return m

    def test_paths_agree(self):
        for algorithm in (0, 1, 2, 3, 4):
            m = self._model()
            r = m.solve(verbosity=0, algorithm=algorithm)
            self.assertTrue(r.optimal, "algorithm %d did not reach optimal" % algorithm)
            self.assertAlmostEqual(r.objective, 36.0, places=6,
                                   msg="algorithm %d disagreed" % algorithm)
            m.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
