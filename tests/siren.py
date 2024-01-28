import unittest
import numpy as np
import torch
from siren_pytorch import Sine

fd_schemes = {
    0: [1.0],
    1: [-0.5, 0, +0.5],
    2: [1.0, -2.0, 1.0],
    3: [-0.5, 1.0, 0.0, -1.0, 0.5],
    4: [1.0, -4.0, 6.0, -4.0, 1.0],
}

class TestSine(unittest.TestCase):
    def test_derivative(self):
        w0 = 8.0
        sine = Sine(w0)

        x = torch.linspace(-np.pi, np.pi, 1024, dtype=torch.float64)
        eps = 1e-4
        for der in range(4):
            actual = sine.forward(x, der=der)
            approx = torch.zeros_like(actual)

            fd = fd_schemes[der]
            for i,coeff in enumerate(fd, -(len(fd)//2)):
                approx += coeff * torch.sin(w0*(x+i*eps))
            approx /= eps**der

            max_err = abs(actual - approx).max()
            print(f'der={der}, err={max_err:2.2e}')
            assert np.allclose(actual, approx, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()
