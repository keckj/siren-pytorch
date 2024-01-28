import unittest
import numpy as np
import torch
from siren_pytorch import Sine, Siren, SirenNet, Sigmoid

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
            print(f'Sine der={der}, err={max_err:2.2e}')
            assert torch.allclose(actual, approx, rtol=1e-3)


class TestSiren(unittest.TestCase):
    def test_derivative(self):
        w0 = 8.0

        dim_in  = 2
        dim_out = 3
        siren = Siren(dim_in, dim_out, w0=w0).double()

        x = torch.rand(1024, dtype=torch.float64)
        y = torch.rand(1024, dtype=torch.float64)

        eps = 1e-6

        z, dz = siren(torch.column_stack([x, y]))

        yp, _ = siren(torch.column_stack([x+eps, y]))
        yn, _ = siren(torch.column_stack([x-eps, y]))
        dz_dx = (yp-yn)/(2*eps)

        yp, _ = siren(torch.column_stack([x, y+eps]))
        yn, _ = siren(torch.column_stack([x, y-eps]))
        dz_dy = (yp-yn)/(2*eps)

        dz_fd = torch.dstack([dz_dx, dz_dy])

        max_err = abs(dz - dz_fd).max()
        print(f'Siren err={max_err:2.2e}')
        assert torch.allclose(dz, dz_fd, rtol=1e-8)


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

        z, dz = siren(torch.column_stack([x, y]))

        yp, _ = siren(torch.column_stack([x+eps, y]))
        yn, _ = siren(torch.column_stack([x-eps, y]))
        dz_dx = (yp-yn)/(2*eps)

        yp, _ = siren(torch.column_stack([x, y+eps]))
        yn, _ = siren(torch.column_stack([x, y-eps]))
        dz_dy = (yp-yn)/(2*eps)

        dz_fd = torch.dstack([dz_dx, dz_dy])

        max_err = abs(dz - dz_fd).max()
        print(f'Siren err={max_err:2.2e}')
        assert torch.allclose(dz, dz_fd, rtol=1e-8)

if __name__ == '__main__':
    unittest.main()
