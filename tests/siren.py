import unittest
import numpy as np
import torch
from siren_pytorch import Sine, Sigmoid, Identity, Siren, SirenNet, Sigmoid

fd_schemes = {
    0: [1.0],
    1: [-0.5, 0, +0.5],
    2: [1.0, -2.0, 1.0],
    3: [-0.5, 1.0, 0.0, -1.0, 0.5],
    4: [1.0, -4.0, 6.0, -4.0, 1.0],
}

class TestActivation(unittest.TestCase):
    def test_sine(self):
        self._test_activation(Sine(30.0))

    def test_sigmoid(self):
        self._test_activation(Sigmoid())
    
    def test_identity(self):
        self._test_activation(Identity())
    
    @classmethod
    def _test_activation(cls, fn):
        fn = fn.double()
        x = torch.linspace(-np.pi, np.pi, 1024, dtype=torch.float64)
        eps = 1e-5
        for der in range(3):
            actual = fn(x, der=der)
            approx = torch.zeros_like(actual)

            fd = fd_schemes[der]
            for i,coeff in enumerate(fd, -(len(fd)//2)):
                approx += coeff * fn(x+i*eps)
            approx /= eps**der

            max_err = abs(actual - approx).max()
            print(f'{type(fn).__name__} der={der}, err={max_err:2.2e}')
            assert torch.allclose(actual, approx, rtol=1e-3, atol=1e-3)


class TestSiren(unittest.TestCase):
    def test_derivative(self):
        w0 = 8.0

        dim_in  = 2
        dim_out = 3

        siren = Siren(dim_in, dim_out, w0=w0).double()

        x = torch.rand(1024, dtype=torch.float64)
        y = torch.rand(1024, dtype=torch.float64)

        eps = 1e-6

        z, J, H = siren(torch.column_stack([x, y]))

        yp, Jp, _ = siren(torch.column_stack([x+eps, y]))
        yn, Jn, _ = siren(torch.column_stack([x-eps, y]))
        dz_dx = (yp-yn)/(2*eps)
        Jz_dx = (Jp-Jn)/(2*eps)

        yp, Jp, _ = siren(torch.column_stack([x, y+eps]))
        yn, Jn, _ = siren(torch.column_stack([x, y-eps]))
        dz_dy = (yp-yn)/(2*eps)
        Jz_dy = (Jp-Jn)/(2*eps)

        Jfd = torch.stack([dz_dx, dz_dy], -1)
        Hfd = torch.stack([Jz_dx, Jz_dy], -1)

        J_max_err = abs(J-Jfd).max()
        H_max_err = abs(H-Hfd).max()
        print(f'Siren Jerr={J_max_err:2.2e}, Herr={H_max_err:2.2e},')
        assert torch.allclose(J, Jfd, rtol=1e-8)
        assert torch.allclose(H, Hfd, rtol=1e-8)


class TestSirenNet(unittest.TestCase):
    def test_derivative(self):
        w0 = 8.0
        w0_initial = 30.0

        dim_in  = 2
        dim_hidden = 32
        dim_out = 3
        num_layers = 4

        siren = SirenNet(dim_in, dim_hidden, dim_out, num_layers,
                w0=w0, w0_initial=w0_initial, final_activation=Sigmoid()).double()

        x = torch.rand(1024, dtype=torch.float64)
        y = torch.rand(1024, dtype=torch.float64)
        eps = 1e-6

        z, J, H = siren(torch.column_stack([x, y]))

        yp, Jp, _ = siren(torch.column_stack([x+eps, y]))
        yn, Jn, _ = siren(torch.column_stack([x-eps, y]))
        dz_dx = (yp-yn)/(2*eps)
        Jz_dx = (Jp-Jn)/(2*eps)

        yp, Jp, _ = siren(torch.column_stack([x, y+eps]))
        yn, Jn, _ = siren(torch.column_stack([x, y-eps]))
        dz_dy = (yp-yn)/(2*eps)
        Jz_dy = (Jp-Jn)/(2*eps)

        Jfd = torch.stack([dz_dx, dz_dy], -1)
        Hfd = torch.stack([Jz_dx, Jz_dy], -1)

        J_max_err = abs(J-Jfd).max()
        H_max_err = abs(H-Hfd).max()
        print(f'Siren Jerr={J_max_err:2.2e}, Herr={H_max_err:2.2e},')
        assert torch.allclose(J, Jfd, rtol=1e-8)
        assert torch.allclose(H, Hfd, rtol=1e-8)

if __name__ == '__main__':
    unittest.main()
