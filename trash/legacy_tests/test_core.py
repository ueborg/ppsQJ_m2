import unittest

import numpy as np

from pps_qj.core.numerics import heff_from, safe_normalize, safe_probs


class TestCore(unittest.TestCase):
    def test_heff_from(self) -> None:
        H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        L = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        heff = heff_from(H, [L])
        expected = H - 0.5j * (L.conj().T @ L)
        self.assertTrue(np.allclose(heff, expected))

    def test_safe_helpers(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.complex128)
        nv = safe_normalize(v)
        self.assertAlmostEqual(float(np.linalg.norm(nv)), 1.0, places=12)

        p = safe_probs(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        self.assertTrue(np.all(p >= 0.0))


if __name__ == "__main__":
    unittest.main()
