"""Tests for Record, Apply, and ResidualSequential."""
import copy
import pytest
import torch
import torch.nn as nn
from torchresidual import Record, Apply, ResidualSequential, LearnableAlpha


# ── Record ────────────────────────────────────────────────────────────────────
class TestRecord:
    def test_forward_passthrough(self):
        r = Record(name="x")
        x = torch.randn(4, 8)
        out = r(x)
        assert torch.equal(out, x)

    def test_stores_tensor_and_shape(self):
        r = Record(name="x")
        x = torch.randn(4, 8)
        r(x)
        assert r.recorded_tensor is not None
        assert r.recorded_shape == x.shape

    def test_auto_naming(self):
        block = ResidualSequential(
            Record(),
            nn.Linear(8, 8),
            Apply(),
        )
        # After construction the record should have been auto-named
        records = [m for m in block if isinstance(m, Record)]
        assert records[0].name is not None

    def test_repr(self):
        r = Record(need_projection=True, name="foo")
        assert "Record" in repr(r)
        assert "foo" in repr(r)


# ── Apply ─────────────────────────────────────────────────────────────────────
class TestApply:
    def test_raises_outside_residual_sequential(self):
        a = Apply(operation="add")
        x = torch.randn(4, 8)
        with pytest.raises(RuntimeError, match="ResidualSequential"):
            a(x)

    def test_invalid_operation_raises(self):
        with pytest.raises(ValueError):
            Apply(operation="unsupported_op")

    def test_repr(self):
        a = Apply(operation="add", record_name="input", alpha=0.5)
        assert "Apply" in repr(a)
        assert "add" in repr(a)


# ── ResidualSequential operations ─────────────────────────────────────────────
def _make_block(operation, dim=16, alpha=1.0):
    return ResidualSequential(
        Record(name="r"),
        nn.Linear(dim, dim),
        Apply(operation=operation, record_name="r", alpha=alpha),
    )


class TestResidualSequentialOperations:
    def test_add(self):
        block = _make_block("add")
        x = torch.randn(4, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_multiply(self):
        block = _make_block("multiply")
        x = torch.randn(4, 16)
        assert block(x).shape == x.shape

    def test_gated(self):
        block = _make_block("gated", alpha=0.5)
        x = torch.randn(4, 16)
        assert block(x).shape == x.shape

    def test_highway(self):
        block = _make_block("highway")
        x = torch.randn(4, 16)
        assert block(x).shape == x.shape

    def test_concat(self):
        block = ResidualSequential(
            Record(name="r"),
            nn.Linear(16, 16),
            Apply(operation="concat", record_name="r"),
        )
        x = torch.randn(4, 16)
        out = block(x)
        assert out.shape == (4, 32)  # concat doubles last dim

    def test_learnable_alpha(self):
        block = _make_block("add", alpha=LearnableAlpha(0.5, 0.0, 1.0))
        x = torch.randn(4, 16)
        out = block(x)
        loss = out.sum()
        loss.backward()
        # Check LearnableAlpha param received a gradient
        for m in block.modules():
            if isinstance(m, LearnableAlpha):
                assert m.param.grad is not None


class TestResidualSequentialMultipleRecords:
    def test_multiple_named_records(self):
        block = ResidualSequential(
            Record(name="in", need_projection=True),
            nn.Linear(32, 16),
            Record(name="mid"),
            nn.Linear(16, 32),
            Apply(record_name="in"),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            Apply(record_name="mid", operation="add"),
        )
        x = torch.randn(4, 32)
        out = block(x)
        assert out.shape == (4, 16)

    def test_no_record_name_uses_latest(self):
        block = ResidualSequential(
            nn.Linear(16, 16),
            Record(),
            nn.Linear(16, 16),
            Apply(),  # should use the single unnamed record
        )
        x = torch.randn(4, 16)
        assert block(x).shape == x.shape


class TestResidualSequentialShapeProjection:
    def test_projection_created_when_needed(self):
        block = ResidualSequential(
            Record(name="r", need_projection=True),
            nn.Linear(16, 32),
            Apply(record_name="r"),
        )
        x = torch.randn(4, 16)
        out = block(x)
        assert out.shape == (4, 32)

    def test_shape_mismatch_raises_without_projection(self):
        block = ResidualSequential(
            Record(name="r", need_projection=False),
            nn.Linear(16, 32),
            Apply(record_name="r"),
        )
        x = torch.randn(4, 16)
        with pytest.raises(RuntimeError, match="need_projection"):
            block(x)


class TestResidualSequentialGradients:
    def test_gradient_flows_through_residual(self):
        block = ResidualSequential(
            Record(name="r"),
            nn.Linear(16, 16),
            nn.ReLU(),
            Apply(record_name="r"),
        )
        x = torch.randn(4, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.norm().item() > 0


class TestResidualSequentialSerialization:
    def test_state_dict_round_trip(self):
        block = ResidualSequential(
            Record(name="r"),
            nn.Linear(16, 16),
            Apply(record_name="r"),
        )
        x = torch.randn(4, 16)
        out1 = block(x)
        sd = block.state_dict()
        block2 = ResidualSequential(
            Record(name="r"),
            nn.Linear(16, 16),
            Apply(record_name="r"),
        )
        block2.load_state_dict(sd)
        out2 = block2(x)
        assert torch.allclose(out1, out2)

    def test_deep_copy(self):
        block = ResidualSequential(
            Record(name="r"),
            nn.Linear(16, 16),
            Apply(record_name="r"),
        )
        block2 = copy.deepcopy(block)
        x = torch.randn(4, 16)
        out1 = block(x)
        out2 = block2(x)
        assert torch.allclose(out1, out2)


class TestResidualSequentialContext:
    def test_nested_residual_sequentials(self):
        inner = ResidualSequential(
            Record(name="inner_r"),
            nn.Linear(16, 16),
            Apply(record_name="inner_r"),
        )
        outer = ResidualSequential(
            Record(name="outer_r"),
            inner,
            Apply(record_name="outer_r"),
        )
        x = torch.randn(4, 16)
        out = outer(x)
        assert out.shape == x.shape

    def test_context_restored_after_forward(self):
        import threading as _threading
        from torchresidual.core import _context

        block = ResidualSequential(
            Record(name="r"),
            nn.Linear(8, 8),
            Apply(record_name="r"),
        )
        prev = getattr(_context, "current_sequential", None)
        block(torch.randn(2, 8))
        assert getattr(_context, "current_sequential", None) is prev

    def test_warn_apply_without_record(self):
        with pytest.warns(UserWarning):
            ResidualSequential(
                nn.Linear(8, 8),
                Apply(),
            )
