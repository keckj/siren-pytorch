import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x, der=0):
        return ((-1)**(der//2))*(self.w0**der)*(torch.sin if der%2==0 else torch.cos)(self.w0 * x)


class Identity(nn.Identity):
    def forward(self, x, der=0):
        if der == 0:
            return super().forward(x)
        elif der == 1:
            return torch.ones_like(x)
        elif der > 1:
            return torch.zeros_like(x)
        else:
            raise NotImplementedError(der)


class Sigmoid(nn.Sigmoid):
    def forward(self, x, der=0):
        if der == 0:
            return super().forward(x)
        elif der == 1:
            y = torch.exp(x)
            return y / (1+y)**2
        elif der == 2:
            y = torch.exp(x)
            return y*(1-y)/ (1+y)**3
        elif der == 3:
            y = torch.exp(x)
            return y*((y-4)*y+1) / (1+y)**4
        elif der == 4:
            y = torch.exp(x)
            return y*(((-y+11)*y-11)*y+1) / (1+y)**5
        else:
            raise NotImplementedError(der)


class SirenOutput:
    def __init__(self, *args):
        """
        Create a new SirenOutput object either from a tuple of torch tensors
        representing function evaluation and successive spatial derivatives
        f, df, d2f, d3f, and so on. Input can also contain None values.
        """
        assert len(args) >= 1, 'SirenOutput requires at least function evaluation.'
        assert all(isinstance(_, (torch.Tensor, type(None))) for _ in args), tuple(map(type, args))
        self.tensors = list(args)

    def resize(self, max_derivative):
        """Resize output to handle at least max_derivative spatial derivatives."""
        assert max_derivative >= 0, max_derivative
        missing_derivatives = max_derivative - len(self) + 1
        if missing_derivatives > 0:
            self.tensors.extend([None] * missing_derivatives)
        assert(len(self) >= max_derivative+1)
        return self

    @property
    def f(self):
        """Evaluated function."""
        return self.tensors[0]

    def compose(self, other):
        """Compose derivatives f(g(x)) where g represent self and f other."""
        assert isinstance(other, SirenOutput), type(other)
        assert len(other) == len(self), (len(other), len(self))
        if not self:
            return other
        D = len(self)
        out = SirenOutput(other.f).resize(D-1)
        for der in range(1, D):
            out[der] = self._compose_derivative(der, other)
        return out

    def _compose_derivative(self, der, other):
        """Compose derivative order der using Faà di Bruno's formulas for chain rule for f(g(x))."""
        if der == 1:    # compose jacobians
            return torch.matmul(other.df, self.df)
        elif der == 2:  # compose hessians
            H0 = torch.einsum('...ij, ...jkl -> ...ikl', other.df, self.d2f)
            H1 = torch.einsum('...ji, ...kjl, ...lp -> ...kip', self.df, other.d2f, self.df)
            return H0 + H1
        elif der == 3:  # compose third order derivatives
            H0 = torch.einsum('...ijkl,...jm,...kn,...lo->...imno', other.d3f, self.df, self.df, self.df)
            H1 = 3*torch.einsum('...ijk,...jl,...kmn->...ilmn', other.d2f, self.df, self.d2f)
            H2 = torch.einsum('...ijkl,...mi->...mjkl', self.d3f, other.df)
            return H0 + H1 + H2
        else:
            raise NotImplementedError(der)

    def __getattr__(self, name):
        """Get other derivatives as df, d2f, d3f, ..."""
        if name.startswith('d') and name.endswith('f'):
            try:
                order = 1 if name=='df' else int(name[1:-1])
                assert order > 0, "Derivative order must be positive"
                assert order < len(self), f"Derivative of order {order} not available, max order is {len(self)-1}."
                return self.tensors[order]
            except ValueError:
                pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, k):
        assert k < len(self), (k, len(self))
        return self.tensors[k]

    def __setitem__(self, k, v):
        assert k < len(self), (k, len(self))
        assert isinstance(v, (torch.Tensor, type(None)))
        self.tensors[k] = v

    def __iter__(self):
        for d in self.tensors:
            yield d

    def __tuple__(self):
        return tuple(self.tensors)

    def __len__(self):
        return len(self.tensors)

    def __bool__(self):
        return all(_ is not None for _ in self)

    def __str__(self):
        def tensor_info(i):
            try:
                return f'f={tuple(self.f.shape)}' if i == 0 else f'd{i}f={tuple(self[i].shape)}'
            except AttributeError:
                return f'f=None' if i == 0 else f'd{i}f=None'
        return f'{self.__class__.__name__}(' + ', '.join(tensor_info(i) for i in range(len(self))) + ')'




class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x, max_derivative=0):
        y = F.linear(x, self.weight, self.bias)
        z = self.activation(y)

        out = SirenOutput(z).resize(max_derivative)

        if max_derivative >= 1:
            out[1] = torch.einsum('...i,ij->...ij',
                        self.activation(y, der=1), self.weight)
        if max_derivative >= 2:
            out[2] = torch.einsum('...i,ij,ik->...ijk',
                        self.activation(y, der=2), self.weight, self.weight)
        if max_derivative >= 3:
            out[3] = torch.einsum('...i,ij,ik,il->...ijkl',
                        self.activation(y, der=3), self.weight, self.weight, self.weight)
        if max_derivative >= 4:
            out[4] = torch.einsum('...i,ij,ik,il,im->...ijklm',
                        self.activation(y, der=4), self.weight, self.weight, self.weight, self.weight)
        if max_derivative >= 5:
            raise NotImplementedError(der)

        return out

# siren network

class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0 = 1.,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
            )

            self.layers.append(layer)

        final_activation = Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, max_derivative=0):
        x = SirenOutput(x).resize(max_derivative)

        for layer in self.layers:
            y = layer(x.f, max_derivative=max_derivative)
            x = x.compose(y)

        y = self.last_layer(x.f, max_derivative=max_derivative)
        return x.compose(y)


# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)

# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(img):
            return F.mse_loss(img, out)

        return out
