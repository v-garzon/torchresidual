"""End-to-end integration tests."""
import torch
import torch.nn as nn
from torchresidual import (
    ResidualSequential,
    Record,
    Apply,
    LearnableAlpha,
    RecurrentWrapper,
)


class TestLSTMIntegration:
    def test_lstm_residual_same_size(self):
        block = ResidualSequential(
            Record(name="r", need_projection=False),
            RecurrentWrapper(
                nn.LSTM(16, 16, batch_first=True),
                return_hidden=False,
            ),
            Apply(record_name="r"),
        )
        x = torch.randn(4, 10, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_lstm_residual_with_projection(self):
        block = ResidualSequential(
            Record(name="r", need_projection=True),
            RecurrentWrapper(
                nn.LSTM(16, 32, batch_first=True),
                return_hidden=False,
            ),
            Apply(record_name="r"),
        )
        x = torch.randn(4, 10, 16)
        out = block(x)
        assert out.shape == (4, 10, 32)

    def test_lstm_return_hidden_extra_output(self):
        block = ResidualSequential(
            Record(name="r", need_projection=False),
            RecurrentWrapper(
                nn.LSTM(16, 16, batch_first=True),
                return_hidden=True,
            ),
            Apply(record_name="r"),
        )
        x = torch.randn(4, 10, 16)
        result = block(x)
        # When return_hidden=True the block returns (output, [hidden])
        assert isinstance(result, tuple)
        assert result[0].shape == x.shape


class TestLearnableAlphaIntegration:
    def test_learnable_alpha_trains(self):
        block = ResidualSequential(
            Record(name="r"),
            nn.Linear(16, 16),
            Apply(record_name="r", operation="gated", alpha=LearnableAlpha(0.5)),
        )
        opt = torch.optim.SGD(block.parameters(), lr=0.01)
        x = torch.randn(4, 16)
        for _ in range(3):
            opt.zero_grad()
            loss = block(x).sum()
            loss.backward()
            opt.step()
        # Just check it doesn't crash and alpha is still in bounds
        for m in block.modules():
            if isinstance(m, LearnableAlpha):
                v = m().item()
                assert 0.0 <= v <= 1.0


class TestComplexArchitecture:
    def test_multi_record_apply_block(self):
        block = ResidualSequential(
            Record(name="input", need_projection=True),
            nn.Linear(64, 32),
            nn.ReLU(),
            Record(name="mid"),
            nn.Linear(32, 64),
            Apply(record_name="input"),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            Apply(record_name="mid", operation="add"),
        )
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == (8, 32)

    def test_gradient_through_multiple_residuals(self):
        block = ResidualSequential(
            Record(name="r1"),
            nn.Linear(16, 16),
            Record(name="r2"),
            nn.Linear(16, 16),
            Apply(record_name="r1"),
            Apply(record_name="r2", operation="multiply"),
        )
        x = torch.randn(4, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
