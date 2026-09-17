"""Independent directional formula, gradient and renderer preflight."""
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from directional_transport import DirectionalTransport, build_transport
from transport import Transport, quadrature_mass


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    model = DirectionalTransport(rank=5).double()
    n, m = 17, 11
    source = dict(means=torch.randn(n, 3, dtype=torch.double),
                  features=torch.randn(n, 32, dtype=torch.double),
                  scales=torch.randn(n, 3, dtype=torch.double)-3,
                  opacities=torch.randn(n, dtype=torch.double))
    g = SimpleNamespace(params=source, center=torch.zeros(3, dtype=torch.double), radius=3.)
    receivers = dict(means=torch.randn(m, 3, dtype=torch.double),
                     features=torch.randn(m, 32, dtype=torch.double),
                     base=torch.randn(m, 3, dtype=torch.double),
                     visibility=torch.rand(m, dtype=torch.double))
    eye = torch.tensor([0., 1., 5.], dtype=torch.double)
    light = torch.tensor([3., 4., 5.], dtype=torch.double)
    intensity = torch.tensor([2., 3., 4.], dtype=torch.double)
    visibility = torch.rand(n, dtype=torch.double)
    with torch.no_grad():
        model.exchange.weight.normal_(std=.2)
    out = model(g, receivers, eye, light, intensity, visibility)
    mass = quadrature_mass(source)
    w = model.partition(source['means']/g.radius).exp()
    a = model.exchange(source['features']).sigmoid()
    delta = light-source['means']
    e = intensity/delta.square().sum(-1, keepdim=True)*visibility[:, None]
    basis = model.direction_basis(source['features'], F.normalize(delta, dim=-1))
    z = torch.zeros(5, 4, 3, dtype=torch.double)
    for r in range(5):
        for c in range(3):
            weights = mass*a[:, c]*w[:, r]
            for b in range(4):
                z[r,b,c] = (weights*basis[:,b]*e[:,c]).sum()/weights.sum()
    y = torch.zeros_like(z)
    for r in range(5):
        for c in range(3):
            y[r,:,c] = F.softplus(model.raw_C[r,c]) @ z[r,:,c]
    h = torch.einsum('mr,rbc->mbc', model.partition(receivers['means']/g.radius).exp(), y)
    bout = model.direction_basis(receivers['features'], F.normalize(eye-receivers['means'], dim=-1))
    aout = model.exchange(receivers['features']).sigmoid()
    direct = model(g, receivers, eye, light, intensity, visibility, False)
    reference = (1-aout)*direct + aout*(bout[:,:,None]*h).sum(1)
    torch.testing.assert_close(out, reference, atol=1e-12, rtol=1e-10)
    old = Transport(rank=5).double()
    old.load_state_dict({k:v for k,v in model.state_dict().items() if k in old.state_dict()})
    torch.testing.assert_close(direct, old(g,receivers,eye,light,intensity,visibility,False))
    torch.testing.assert_close(model(g,receivers,eye,light,intensity*3,visibility), out*3)
    dark = dict(receivers, visibility=torch.zeros(m, dtype=torch.double))
    torch.testing.assert_close(model(g,dark,eye,light,intensity,visibility), aout*(bout[:,:,None]*h).sum(1))
    out.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    print('PASS: independent formula, legacy disabled ports, light linearity, shadowed receiver, all parameter gradients')

    from data import SceneDataset
    from gaussians import Gaussians, camera_bounds
    from evaluate import to_device, render_observation
    sample = to_device(SceneDataset('/workspace/datasets/SSD-GS/data/Real_NRHints/Cat', 'train', 64)[0], 'cuda')
    center, radius = camera_bounds([sample])
    g = Gaussians(256, center, radius, 32)
    model = DirectionalTransport().cuda()
    image, alpha, _ = render_observation(g, model, sample, 0., True, True, 2.2, 'deep')
    (image.mean()+alpha.mean()).backward()
    assert torch.isfinite(image).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    print('PASS: CUDA deep-shadow pixel renderer forward/backward')


if __name__ == '__main__':
    main()
