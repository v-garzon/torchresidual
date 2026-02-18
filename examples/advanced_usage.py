"""Advanced usage examples for torchresidual."""
import torch
import torch.nn as nn
from collections import OrderedDict
from torchresidual import (
    ResidualSequential,
    Record,
    Apply,
    LearnableAlpha,
    RecurrentWrapper,
)


# ── Example 1: Multiple skip connections with different operations ──────────
print("Example 1: Multiple skip connections")
print("=" * 60)

multi_skip = ResidualSequential(
    Record(name="input", need_projection=True),
    nn.Linear(64, 128),
    nn.ReLU(),
    Record(name="mid"),
    nn.Linear(128, 128),
    nn.ReLU(),
    Apply(record_name="input", operation="add"),  # Long skip with projection
    nn.LayerNorm(128),
    nn.Linear(128, 64),
    Apply(record_name="mid", operation="concat"),  # Concat doubles last dim
)

x = torch.randn(4, 64)
out = multi_skip(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")  # [4, 192] due to concat
print()


# ── Example 2: Learnable alpha with training loop ───────────────────────────
print("Example 2: Training with learnable alpha")
print("=" * 60)

trainable_block = ResidualSequential(
    Record(name="r"),
    nn.Linear(32, 32),
    nn.ReLU(),
    Apply(
        record_name="r",
        operation="gated",
        alpha=LearnableAlpha(0.5, min_value=0.0, max_value=1.0),
    ),
)

optimizer = torch.optim.Adam(trainable_block.parameters(), lr=1e-3)

# Dummy training loop
for epoch in range(5):
    x = torch.randn(8, 32)
    target = torch.randn(8, 32)
    
    optimizer.zero_grad()
    out = trainable_block(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    
    # Extract alpha value
    for module in trainable_block.modules():
        if isinstance(module, LearnableAlpha):
            alpha_val = module().item()
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, alpha={alpha_val:.4f}")
            break

print()


# ── Example 3: Encoder-decoder with skip connections ────────────────────────
print("Example 3: Encoder-decoder architecture")
print("=" * 60)

encoder_decoder = ResidualSequential(
    OrderedDict([
        # Encoder
        ('enc_record1', Record(name="enc1", need_projection=True)),
        ('enc_conv1', nn.Conv1d(16, 32, kernel_size=3, padding=1)),
        ('enc_relu1', nn.ReLU()),
        
        ('enc_record2', Record(name="enc2", need_projection=True)),
        ('enc_conv2', nn.Conv1d(32, 64, kernel_size=3, padding=1)),
        ('enc_relu2', nn.ReLU()),
        
        # Bottleneck
        ('bottleneck', nn.Conv1d(64, 64, kernel_size=3, padding=1)),
        
        # Decoder
        ('dec_conv1', nn.Conv1d(64, 32, kernel_size=3, padding=1)),
        ('dec_relu1', nn.ReLU()),
        ('dec_apply1', Apply(record_name="enc2", operation="add")),
        
        ('dec_conv2', nn.Conv1d(32, 16, kernel_size=3, padding=1)),
        ('dec_relu2', nn.ReLU()),
        ('dec_apply2', Apply(record_name="enc1", operation="add")),
    ])
)

x = torch.randn(4, 16, 100)  # [batch, channels, sequence]
out = encoder_decoder(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print()


# ── Example 4: LSTM with bidirectional and projection ───────────────────────
print("Example 4: Bidirectional LSTM with projection")
print("=" * 60)

lstm_block = ResidualSequential(
    Record(name="r", need_projection=True),
    RecurrentWrapper(
        nn.LSTM(32, 64, num_layers=2, bidirectional=True, batch_first=True),
        return_hidden=False,
    ),
    nn.LayerNorm(128),  # 64 * 2 (bidirectional)
    nn.Linear(128, 32),
    Apply(record_name="r", operation="add"),
)

x = torch.randn(4, 10, 32)  # [batch, seq_len, features]
out = lstm_block(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print()


# ── Example 5: Highway network block ────────────────────────────────────────
print("Example 5: Highway network")
print("=" * 60)

highway_block = ResidualSequential(
    Record(name="r"),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.LayerNorm(64),
    Apply(record_name="r", operation="highway"),  # Learnable gates
)

x = torch.randn(4, 64)
out = highway_block(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")

# Highway creates two gates (transform and carry)
print("Highway gates created:")
for name, module in highway_block.named_modules():
    if isinstance(module, Apply) and module.gate_layer is not None:
        print(f"  Transform gate: {module.gate_layer['transform_gate']}")
        print(f"  Carry gate: {module.gate_layer['carry_gate']}")
        break

print()


# ── Example 6: Nested residual blocks ───────────────────────────────────────
print("Example 6: Nested residual blocks")
print("=" * 60)

inner_block = ResidualSequential(
    Record(name="inner"),
    nn.Linear(64, 64),
    nn.ReLU(),
    Apply(record_name="inner"),
)

outer_block = ResidualSequential(
    Record(name="outer"),
    inner_block,
    nn.LayerNorm(64),
    nn.Linear(64, 64),
    Apply(record_name="outer"),
)

x = torch.randn(4, 64)
out = outer_block(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print("Inner and outer residual connections both applied")
print()


# ── Example 7: Log-space alpha for learning rates ───────────────────────────
print("Example 7: Log-space alpha (for learning rate search)")
print("=" * 60)

# Auto-detects log space when range > 100x
lr_alpha = LearnableAlpha(1e-3, min_value=1e-5, max_value=1e-1)

print(f"Initial value: {lr_alpha().item():.6f}")
print(f"Using log space: {lr_alpha.use_log_space}")
print(f"Min bound: {lr_alpha.min_val.item():.6f}")
print(f"Max bound: {lr_alpha.max_val.item():.6f}")



# ── Example 8: One record, multiple applies ─────────────────────────────────
print("Example 8: One record, multiple applies")
block = ResidualSequential(
    Record(name="x", need_projection=True),
    nn.Linear(16, 16),
    nn.ReLU(),
    Apply(record_name="x", operation="add"),
    nn.Linear(16, 16),
    nn.ReLU(),
    Apply(record_name="x", operation="add"),
)

x = torch.randn(4, 16)
out = block(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print("Multiple applies to the same record work correctly")
print()


print("=" * 60)
print("All examples completed successfully!")
