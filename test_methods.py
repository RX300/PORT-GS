"""CPU operator regressions for selectable transport methods."""
import argparse
import io
import unittest

import torch
from torch.nn import functional as F

from gaussians import Gaussians
from methods import METHODS, add_method_arguments, build_transport, resolve_config
from methods.base import quadrature_mass


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.g = Gaussians(17, torch.zeros(3), 1., 32, device='cpu').double()
        p = self.g.params
        self.receivers = dict(means=p['means'][:11], features=p['features'][:11],
                              base=p['base'][:11], visibility=torch.linspace(.1, 1, 11).double())
        self.eye = torch.tensor([0., 1., 4.], dtype=torch.double)
        self.light = torch.tensor([2., 3., 4.], dtype=torch.double)
        self.intensity = torch.tensor([1., 2., 3.], dtype=torch.double)
        self.visibility = torch.linspace(.2, 1, 17).double()

    def model(self, name):
        torch.manual_seed(7)
        config = resolve_config(dict(representation=name, rank=5))
        return build_transport(config, 1.).double(), config

    def shade(self, model, active=True, intensity=None, receivers=None):
        return model(self.g, self.receivers if receivers is None else receivers,
                     self.eye, self.light, self.intensity if intensity is None else intensity,
                     self.visibility, active)

    def test_method_selection_and_config_roundtrip(self):
        for name in METHODS:
            with self.subTest(name=name):
                argv = ['--representation', name, '--rank', '5']
                if name == 'local_frame':
                    argv += ['--frame-width', '7']
                parser = argparse.ArgumentParser()
                add_method_arguments(parser, argv)
                config = resolve_config(vars(parser.parse_args(argv)))
                self.assertEqual(config['representation'], name)
                model = build_transport(config, 1.).double()
                buffer = io.BytesIO()
                torch.save(dict(config=config, transport=model.state_dict()), buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=False)
                restored = build_transport(saved['config'], 1.).double()
                restored.load_state_dict(saved['transport'], strict=True)
                torch.testing.assert_close(self.shade(model), self.shade(restored), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            build_transport(dict(representation='missing_method'), 1.)

    def test_initial_methods_reduce_to_directional_baseline(self):
        baseline, _ = self.model('directional_port_v1')
        for name in ('paired_port', 'local_frame'):
            model, _ = self.model(name)
            for active in (False, True):
                torch.testing.assert_close(self.shade(model, active), self.shade(baseline, active), rtol=0, atol=0)

    def test_paired_support_and_independent_formula(self):
        model, _ = self.model('paired_port')
        old, _ = self.model('directional_port_v1')
        with torch.no_grad():
            model.output_centers[0].add_(torch.tensor([.3, -.2, .1]))
            model.output_log_width[1].add_(.4)
            model.exchange.weight.normal_(std=.2)
        p = self.g.params
        source_w = model.partition(p['means']).exp()
        receiver_w = F.softmax(-((self.receivers['means'][:, None] - model.output_centers) ** 2).sum(-1)
                               / (2 * model.output_log_width.exp().square()), dim=-1)
        delta = self.light - p['means']
        incident = self.intensity / delta.square().sum(-1, keepdim=True) * self.visibility[:, None]
        incoming = model.direction_basis(p['features'], F.normalize(delta, dim=-1))
        outgoing = model.direction_basis(self.receivers['features'], F.normalize(self.eye-self.receivers['means'], dim=-1))
        mass = quadrature_mass(p)
        fraction = model.exchange(p['features']).sigmoid()
        result = torch.zeros(11, 3, dtype=torch.double)
        for r in range(5):
            for c in range(3):
                weights = mass * fraction[:, c] * source_w[:, r]
                z = (weights[:, None] * incoming * incident[:, c, None]).sum(0) / weights.sum()
                y = F.softplus(model.raw_C[r, c]) @ z
                result[:, c] += receiver_w[:, r] * (outgoing * y).sum(-1)
        a = model.exchange(self.receivers['features']).sigmoid()
        expected = (1-a) * self.shade(model, False) + a * result
        torch.testing.assert_close(self.shade(model), expected, rtol=1e-11, atol=1e-13)
        self.assertGreater(float((self.shade(model)-self.shade(old)).abs().max()), 1e-10)
        torch.testing.assert_close(model.partition(p['means']), old.partition(p['means']), rtol=0, atol=0)
        self.shade(model).square().sum().backward()
        for parameter in (model.anchor_centers, model.log_width, model.output_centers, model.output_log_width):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_frame_rotates_direct_only(self):
        model, _ = self.model('local_frame')
        baseline, _ = self.model('directional_port_v1')
        with torch.no_grad():
            model.frame_net[-1].bias.copy_(torch.tensor([.8, .6, 0., -.6, .8, .2]))
        q = model.shading_frame(self.receivers['features'])
        identity = torch.eye(3, dtype=torch.double).expand(11, -1, -1)
        torch.testing.assert_close(q.transpose(1, 2) @ q, identity, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(torch.linalg.det(q), torch.ones(11, dtype=torch.double))
        self.assertGreater(float((self.shade(model, False)-self.shade(baseline, False)).abs().max()), 1e-10)
        dark = dict(self.receivers, visibility=torch.zeros(11, dtype=torch.double))
        torch.testing.assert_close(self.shade(model, receivers=dark), self.shade(baseline, receivers=dark), rtol=0, atol=0)

    def test_zero_light_linearity_and_optimization(self):
        for name in METHODS:
            with self.subTest(name=name):
                model, _ = self.model(name)
                out = self.shade(model)
                torch.testing.assert_close(self.shade(model, intensity=self.intensity*3), out*3)
                torch.testing.assert_close(self.shade(model, intensity=self.intensity*0), torch.zeros_like(out), rtol=0, atol=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                before = {k: v.detach().clone() for k, v in model.named_parameters()}
                for _ in range(3):
                    optimizer.zero_grad()
                    self.shade(model).square().mean().backward()
                    for key, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, key)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                    optimizer.step()
                changed = [key for key, p in model.named_parameters() if not torch.equal(p, before[key])]
                if name == 'local_frame':
                    self.assertIn('frame_net.0.weight', changed)
                    self.assertIn('frame_net.2.weight', changed)
                elif name == 'paired_port':
                    self.assertIn('output_centers', changed)
                    self.assertIn('output_log_width', changed)


if __name__ == '__main__':
    unittest.main()
