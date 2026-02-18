# Frequently Asked Questions

## General

### Q: Why use torchresidual instead of manual residual connections?

**A:** Manual residual connections work fine for simple cases, but become unwieldy with:
- Multiple skip connections at different depths
- Shape mismatches requiring projection
- Different residual operations (concat, gated, highway)
- Learnable mixing coefficients

`torchresidual` handles all of this with a clean API.

### Q: Does torchresidual add overhead?

**A:** Minimal. The `Record`/`Apply` pattern is essentially a named save/restore operation. 
Thread-local context lookup adds <0.01% overhead. Most time is spent in actual layer computation.

### Q: Can I mix torchresidual with regular nn.Sequential?

**A:** Yes! You can nest them:

```python
regular_block = nn.Sequential(nn.Linear(64, 64), nn.ReLU())

residual_block = ResidualSequential(
    Record(),
    regular_block,  # Works fine
    Apply(),
)
```

---

## Operations

### Q: What's the difference between "gated" and "highway"?

**A:** 
- **Gated** (`operation="gated"`): Single learnable scalar α
  - Formula: `(1-α)·x + α·residual`
  - Interpolates between transformed and residual paths
  
- **Highway** (`operation="highway"`): Two learnable gates (transform T, carry C)
  - Formula: `T·x + C·residual`
  - More expressive but more parameters

### Q: When should I use "concat" instead of "add"?

**A:**
- **Add**: When you want to preserve dimensionality (like ResNet)
- **Concat**: When you want to keep both paths' information (like DenseNet)
  - Warning: concat doubles the last dimension

### Q: Can I use multiple Apply modules with the same Record?

**A:** Yes! Common pattern:

```python
ResidualSequential(
    Record(name="r"),
    layer1,
    Apply(record_name="r"),  # First use
    layer2,
    Apply(record_name="r"),  # Second use (same record)
)
```

---

## LearnableAlpha

### Q: When does LearnableAlpha use log space?

**A:** Automatically when `max_value / min_value > 100` and `min_value > 0`.

**Example:**
```python
# Uses log space (ratio = 1000)
alpha = LearnableAlpha(0.01, min_value=0.001, max_value=1.0)

# Uses linear space (ratio = 10)
alpha = LearnableAlpha(0.5, min_value=0.0, max_value=1.0)
```

**Override:** Set `use_log_space=True/False` explicitly.

### Q: Why tanh instead of sigmoid?

**A:** Tanh provides:
- Better gradient flow near boundaries
- Symmetric parameterization
- Slightly more stable training

Difference is minor in practice.

### Q: Can I use LearnableAlpha outside ResidualSequential?

**A:** Yes! It's a standalone module:

```python
alpha = LearnableAlpha(0.5, 0.0, 1.0)
output = base_output * alpha() + bonus * (1 - alpha())
```

---

## Architecture

### Q: Why thread-local storage instead of direct parent reference?

**A:** Direct parent references create circular refs:
```
ResidualSequential → Apply → ResidualSequential  # Breaks pickle/deepcopy
```

Thread-local storage avoids this while remaining thread-safe for `nn.DataParallel`.

See [DESIGN.md](DESIGN.md) for details.

### Q: Does torchresidual work with torch.jit.script?

**A:** Not in v0.1.0. TorchScript support is planned for v1.1 via a separate 
`ResidualSequentialScript` class.

**Workaround:** Train with `ResidualSequential`, then manually reconstruct as 
`nn.Sequential` for export.

---

## Debugging

### Q: Error: "Apply must be used inside ResidualSequential"

**A:** You called `Apply.forward()` directly. `Apply` only works inside `ResidualSequential`:

```python
# ❌ Wrong
apply = Apply()
apply(x)  # Error!

# ✅ Correct
block = ResidualSequential(Record(), ..., Apply())
block(x)
```

### Q: Error: "Shape mismatch: recorded {shape1} vs current {shape2}"

**A:** Enable automatic projection:

```python
Record(need_projection=True)  # Add this flag
```

Or ensure shapes match manually.

### Q: My LearnableAlpha doesn't seem to be training

**A:** Check that:
1. Alpha is part of the optimizer: `optimizer = Adam(block.parameters())`
2. You're calling `alpha()` not `alpha` in forward pass
3. Loss has gradients flowing through the alpha branch

Debug:
```python
for module in block.modules():
    if isinstance(module, LearnableAlpha):
        print(f"Alpha: {module().item():.4f}, grad: {module.param.grad}")
```

---

## Performance

### Q: Can I use ResidualSequential with DataParallel?

**A:** Yes! Thread-local storage makes it safe:

```python
model = ResidualSequential(...)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

Each GPU thread gets its own context.

### Q: What about DistributedDataParallel?

**A:** Also works, but thread-local storage is unnecessary (each process has 
separate memory). Still safe to use.

### Q: Does LSTM + residual slow down training?

**A:** Negligible. The residual add is ~0.01% of LSTM compute time.

---

## Use Cases

### Q: Can I build a Transformer with torchresidual?

**A:** Yes:

```python
transformer_block = ResidualSequential(
    Record(name="attn_in"),
    nn.MultiheadAttention(embed_dim=512, num_heads=8),
    Apply(record_name="attn_in"),
    nn.LayerNorm(512),
    
    Record(name="ffn_in"),
    nn.Linear(512, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
    Apply(record_name="ffn_in"),
    nn.LayerNorm(512),
)
```

### Q: Can I replicate ResNet-50 architecture?

**A:** Yes, but standard PyTorch's `torchvision.models.resnet50` is more 
optimized for that specific architecture. Use `torchresidual` when you need:
- Custom residual patterns
- Multiple skip connections
- Learnable alpha
- Non-standard operations (gated, highway, concat)

---

## Still have questions?

- Open an issue: https://github.com/v-garzon/torchresidual/issues
- Check examples: [examples/](../examples/)
- Read the source: The entire library is ~500 lines
