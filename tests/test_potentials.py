from __future__ import annotations

import unittest

import numpy as np

from two_timescale.potentials import LocalPCAGeometry, NamedPotential, as_potential


class PotentialTests(unittest.TestCase):
    def test_named_potentials_are_callable(self) -> None:
        geometry = LocalPCAGeometry(
            x_c=np.array([2.0, 1.0]),
            y=np.array([2.0]),
            eigenvalue=np.array([4.0]),
            sigma_sqr=1.0,
            activity=0.5,
            pca_dimensionality=1,
            data_dimensionality=2,
        )

        for name in ("AR", "N", "H", "AV", "VRV", "VRR"):
            value = NamedPotential(name)(geometry)

            self.assertTrue(np.isfinite(value), name)
            self.assertGreater(value, 0.0, name)

    def test_callable_potential_is_wrapped_with_custom_name(self) -> None:
        geometry = LocalPCAGeometry(
            x_c=np.array([2.0, 1.0]),
            y=np.array([2.0]),
            eigenvalue=np.array([4.0]),
            sigma_sqr=1.0,
            activity=1.0,
            pca_dimensionality=1,
            data_dimensionality=2,
        )

        custom = as_potential(lambda unit_geometry: unit_geometry.activity + unit_geometry.y.sum())

        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom(geometry), 3.0)


if __name__ == "__main__":
    unittest.main()
