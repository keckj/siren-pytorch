import unittest
import functools
import numpy as np
import torch
from siren_pytorch import Sine, Sigmoid, Identity, Siren, SirenNet, Sigmoid

device = 'cuda:0' if torch.cuda.is_available else 'cpu'
print(f"Using device {device}.")

class TestActivation(unittest.TestCase):
    def test_sine(self):
        self._test_activation(Sine(2.0))

    def test_sigmoid(self):
        self._test_activation(Sigmoid())

    def test_identity(self):
        self._test_activation(Identity())

    @classmethod
    def _test_activation(cls, fn):
        fn = fn.double().to(device)
        x = torch.linspace(-np.pi, np.pi, 1024, dtype=torch.float64).to(device)
        eps = 1e-6
        for der in range(1,5):
            actual = fn(x, der=der)
            approx = (fn(x+eps, der=der-1) - fn(x-eps, der=der-1)) / (2*eps)

            max_err = abs(actual - approx).max()
            print(f'Activation func {type(fn).__name__} der={der}, err={max_err:2.2e}')
            assert torch.allclose(actual, approx, rtol=1e-8, atol=1e-8)


class TestSiren(unittest.TestCase):
    def test_derivative(self):
        w0 = 8.0

        dim_in  = 2
        dim_out = 3
        max_derivative = 4

        siren = Siren(dim_in, dim_out, w0=w0).double().to(device)
        def f(x, y):
            return siren(torch.column_stack([x, y]), max_derivative=max_derivative)

        x = torch.rand(1024, dtype=torch.float64).to(device)
        y = torch.rand(1024, dtype=torch.float64).to(device)

        eps = 1e-6

        z = f(x, y)

        zxp = f(x+eps, y)
        zxn = f(x-eps, y)
        zyp = f(x, y+eps)
        zyn = f(x, y-eps)

        for i in range(1, max_derivative+1):
            ev = z[i]
            fd = torch.stack([zxp[i-1]-zxn[i-1], zyp[i-1]-zyn[i-1]], -1) / (2*eps)

            max_err = abs(ev-fd).max()
            print(f'Siren d{i}f={max_err:2.2e}')
            assert torch.allclose(ev, fd, rtol=1e-8)


class TestSirenNet(unittest.TestCase):
    def test_derivative(self):
        w0 = 8.0
        w0_initial = 30.0

        dim_in  = 2
        dim_hidden = 32
        dim_out = 3
        num_layers = 4

        siren = SirenNet(dim_in, dim_hidden, dim_out, num_layers,
                w0=w0, w0_initial=w0_initial, final_activation=Sigmoid()).double().to(device)
        def f(x, y):
            return siren(torch.column_stack([x, y]), max_derivative=2)

        x = torch.rand(1024, dtype=torch.float64).to(device)
        y = torch.rand(1024, dtype=torch.float64).to(device)
        eps = 1e-6

        z, J, H = f(x, y)
        assert z.shape == (1024, dim_out), z.shape
        assert J.shape == (1024, dim_out, dim_in), J.shape
        assert H.shape == (1024, dim_out, dim_in, dim_in), H.shape

        yp, Jp, Hp = f(x+eps, y)
        yn, Jn, Hn = f(x-eps, y)
        dz_dx = (yp-yn)/(2*eps)
        Jz_dx = (Jp-Jn)/(2*eps)

        yp, Jp, Hp = f(x, y+eps)
        yn, Jn, Hn = f(x, y-eps)
        dz_dy = (yp-yn)/(2*eps)
        Jz_dy = (Jp-Jn)/(2*eps)

        Jfd = torch.stack([dz_dx, dz_dy], -1)
        Hfd = torch.stack([Jz_dx, Jz_dy], -1)

        J_max_err = abs(J-Jfd).max()
        H_max_err = abs(H-Hfd).max()
        print(f'Siren Jerr={J_max_err:2.2e}, Herr={H_max_err:2.2e},')
        assert torch.allclose(J, Jfd, rtol=1e-8)
        assert torch.allclose(H, Hfd, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
