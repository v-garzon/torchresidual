"""Tests for RecurrentWrapper."""
import pytest
import torch
import torch.nn as nn
from torchresidual import RecurrentWrapper


class TestRecurrentWrapper:
    def test_lstm_no_hidden_return_output_only(self):
        wrapper = RecurrentWrapper(nn.LSTM(16, 16, batch_first=True), return_hidden=False)
        x = torch.randn(4, 10, 16)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 10, 16)

    def test_lstm_return_hidden(self):
        wrapper = RecurrentWrapper(nn.LSTM(16, 16, batch_first=True), return_hidden=True)
        x = torch.randn(4, 10, 16)
        out = wrapper(x)
        assert isinstance(out, tuple)
        assert out[0].shape == (4, 10, 16)

    def test_lstm_with_hidden_state(self):
        lstm = nn.LSTM(16, 16, batch_first=True)
        wrapper = RecurrentWrapper(lstm, return_hidden=True)
        x = torch.randn(4, 10, 16)
        out, (h, c) = wrapper(x)
        # Use the returned hidden state in the next call
        out2, _ = wrapper(x, hidden=(h, c))
        assert out2.shape == (4, 10, 16)

    def test_gru_support(self):
        wrapper = RecurrentWrapper(nn.GRU(8, 8, batch_first=True), return_hidden=False)
        x = torch.randn(2, 5, 8)
        out = wrapper(x)
        assert out.shape == (2, 5, 8)

    def test_repr(self):
        wrapper = RecurrentWrapper(nn.LSTM(8, 8), return_hidden=False)
        assert "RecurrentWrapper" in repr(wrapper)
