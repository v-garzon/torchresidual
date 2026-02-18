"""LSTM residual block example."""
import torch
import torch.nn as nn
from torchresidual import ResidualSequential, Record, Apply, RecurrentWrapper

block = ResidualSequential(
    Record(name="r", need_projection=False),
    RecurrentWrapper(
        nn.LSTM(32, 32, num_layers=2, batch_first=True),
        return_hidden=False,
    ),
    Apply(record_name="r", operation="add"),
)

x = torch.randn(4, 10, 32)  # [batch, seq_len, features]
out = block(x)
print(f"LSTM residual: input={x.shape}  output={out.shape}")
