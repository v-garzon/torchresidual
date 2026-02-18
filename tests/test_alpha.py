"""Tests for LearnableAlpha."""
import copy
import math
import pytest
import torch
from torchresidual import LearnableAlpha


class TestLearnableAlphaLinear:
    def test_initial_value_preserved(self):
        alpha = LearnableAlpha(0.5, 0.0, 1.0)
        assert abs(alpha().item() - 0.5) < 1e-4

    def test_bounds_respected_after_large_gradient(self):
        alpha = LearnableAlpha(0.5, 0.0, 1.0)
        # Force param to extreme values
        with torch.no_grad():
            alpha.param.fill_(1e6)
        assert alpha().item() <= 1.0
        with torch.no_grad():
            alpha.param.fill_(-1e6)
        assert alpha().item() >= 0.0

    def test_auto_selects_linear_space(self):
        alpha = LearnableAlpha(0.5, 0.0, 1.0)
        assert not alpha.use_log_space

    def test_gradient_flows(self):
        alpha = LearnableAlpha(0.5, 0.0, 1.0)
        x = torch.randn(4, 8)
        out = x * alpha()
        out.sum().backward()
        assert alpha.param.grad is not None

    def test_arithmetic_operators(self):
        alpha = LearnableAlpha(0.5, 0.0, 1.0)
        v = alpha().item()
        x = torch.tensor(2.0)
        assert abs((alpha * x).item() - v * 2.0) < 1e-5
        assert abs((x * alpha).item() - v * 2.0) < 1e-5
        assert abs((alpha + x).item() - (v + 2.0)) < 1e-5
        assert abs((x + alpha).item() - (v + 2.0)) < 1e-5
        assert abs((alpha - x).item() - (v - 2.0)) < 1e-5
        assert abs((x - alpha).item() - (2.0 - v)) < 1e-5
        assert abs((alpha / x).item() - v / 2.0) < 1e-5
        assert abs((x / alpha).item() - 2.0 / v) < 1e-5

    def test_repr_contains_key_fields(self):
        alpha = LearnableAlpha(0.5, 0.0, 1.0)
        r = repr(alpha)
        assert "LearnableAlpha" in r
        assert "linear" in r
        assert "bounds" in r

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            LearnableAlpha(0.5, 1.0, 0.0)
        with pytest.raises(ValueError):
            LearnableAlpha(0.5, 0.5, 0.5)

    def test_deep_copy(self):
        alpha = LearnableAlpha(0.3, 0.0, 1.0)
        alpha2 = copy.deepcopy(alpha)
        assert abs(alpha2().item() - alpha().item()) < 1e-6

    def test_state_dict_round_trip(self):
        alpha = LearnableAlpha(0.3, 0.0, 1.0)
        sd = alpha.state_dict()
        alpha2 = LearnableAlpha(0.9, 0.0, 1.0)
        alpha2.load_state_dict(sd)
        assert abs(alpha2().item() - alpha().item()) < 1e-6


class TestLearnableAlphaLog:
    def test_auto_selects_log_space(self):
        alpha = LearnableAlpha(0.01, 0.001, 10.0)
        assert alpha.use_log_space

    def test_initial_value_preserved_log(self):
        alpha = LearnableAlpha(0.1, 0.001, 10.0)
        assert abs(alpha().item() - 0.1) < 1e-3

    def test_bounds_respected_log(self):
        alpha = LearnableAlpha(0.1, 0.001, 10.0)
        with torch.no_grad():
            alpha.param.fill_(1e6)
        assert alpha().item() <= 10.0 + 1e-4
        with torch.no_grad():
            alpha.param.fill_(-1e6)
        assert alpha().item() >= 0.001 - 1e-4

    def test_force_log_space(self):
        alpha = LearnableAlpha(0.5, 0.0, 1.0, use_log_space=False)
        assert not alpha.use_log_space
