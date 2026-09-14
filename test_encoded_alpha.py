"""Foreground composition and gradient checks for PNG observations."""

import unittest

import torch

from evaluate import observation_image


class TestEncodedAlpha(unittest.TestCase):
    def test_zero_alpha_is_background(self):
        linear = torch.tensor([[[0.1, 0.3, 0.7]]])
        alpha = torch.zeros(1, 1, 1)
        actual = observation_image(linear, 2.2, alpha=alpha, background=0.25)
        expected = torch.full_like(linear, 0.25)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gamma_one_recovers_valid_composite(self):
        foreground = torch.tensor(
            [[[0.1, 0.5, 1.2], [0.3, 0.8, 0.05]]], dtype=torch.float32
        )
        alpha = torch.tensor([[[0.0], [0.35]]])
        background = 0.2
        composite = foreground * alpha + background * (1 - alpha)
        actual = observation_image(composite, 1.0, alpha=alpha, background=background)
        torch.testing.assert_close(actual, composite, rtol=1e-6, atol=1e-7)

    def test_known_foreground_composition(self):
        foreground = torch.tensor([[[0.1, 0.5, 1.2], [0.3, 0.8, 0.05]]])
        alpha = torch.tensor([[[0.2], [0.75]]])
        background = 0.2
        composite = foreground * alpha + background * (1 - alpha)
        expected = observation_image(foreground, 2.2) * alpha + background * (1 - alpha)
        actual = observation_image(composite, 2.2, alpha=alpha, background=background)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_edge_gradients_are_finite(self):
        linear = torch.tensor(
            [[[0.05, 0.2, 0.7]], [[0.4, 0.6, 1.0]], [[0.1, 0.3, 0.9]]],
            requires_grad=True,
        )
        alpha = torch.tensor([[[0.0]], [[1e-8]], [[1.0]]], requires_grad=True)
        output = observation_image(linear, 2.2, alpha=alpha, background=0.25)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(linear.grad).all())
        self.assertTrue(torch.isfinite(alpha.grad).all())


if __name__ == "__main__":
    unittest.main()
