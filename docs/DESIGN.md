# Design Decisions

## Why thread-local storage?

`Apply` needs to locate the tensor recorded by `Record` earlier in the same
forward pass.  The simplest approach would be to store a reference to the
parent `ResidualSequential` on each `Apply` instance—but that creates a
circular reference:

```
ResidualSequential → Apply → ResidualSequential (parent ref)
```

Circular references complicate garbage collection, break `pickle` / `torch.save`,
and cause subtle bugs with `copy.deepcopy`.

Instead, `torchresidual` uses `threading.local()`:

```python
_context = threading.local()

# inside ResidualSequential.forward():
_context.current_sequential = self   # set for this thread only
```

Each thread maintains its own `_context`, so concurrent execution via
`nn.DataParallel` is safe without locks.

### Alternatives considered

| Approach | Problem |
|----------|---------|
| Parent reference | Circular ref → breaks pickle, deepcopy |
| weakref | Slightly weaker thread-safety guarantees |
| Passing context as argument | Breaks `nn.Module.forward(x)` contract |
| Global registry | Shared mutable state, hard to test |

`threading.local()` is the same pattern used by Flask's request context,
SQLAlchemy's scoped session, and PyTorch's autograd internals.

---

## Why tanh reparameterization for LearnableAlpha?

A learnable scalar constrained to `[a, b]` could use:

* **sigmoid** — maps `(-∞, ∞) → (0, 1)`, then rescale.  Works, but gradient
  saturates heavily near 0 and 1.
* **tanh** — maps `(-∞, ∞) → (-1, 1)`, then rescale.  Slightly better
  gradient flow for values near the boundaries.
* **clamp + STE** — non-differentiable at boundaries.

We chose **tanh** for its smooth gradient profile.

---

## Why log-space auto-detection?

When `max / min > 100` (e.g. learning rates `1e-4` to `1e-1`), equal steps in
linear space explore the lower end of the range very coarsely.  Log-space
gives uniform coverage in log scale, which is usually what the user wants.

Auto-detection threshold of `100×` is a heuristic; users can override with
`use_log_space=True/False`.

---

## TorchScript / ONNX (planned v1.1)

`threading.local()` is a Python runtime object and cannot be serialised by
TorchScript's IR.  A JIT-compatible sibling class (`ResidualSequentialScript`)
that pre-computes the Record→Apply index map at `__init__` time is planned for
v1.1, once there is demonstrated user demand.
