"""
Test Suite: Strict Causal Invariance Verification for Model 5 (CNN + LSTM)
==========================================================================
Verifies that:
1. Future inputs (at step >= t) cannot alter predictions at earlier steps (< t).
2. Gradients of output at step t with respect to future inputs (step > t) are identically zero.
3. Sequence length is strictly preserved through the causal convolution and recurrent layer.
4. Feature alignment guarantees that next-event correctness does not leak into earlier predictions.
"""

import sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models import CNNLSTM


def test_causal_convolution_invariance():
    torch.manual_seed(42)
    model = CNNLSTM(
        num_questions=1000,
        num_parts=8,
        num_tags=100,
        embed_dim=32,
        hidden_dim=32,
        kernel_size=3,
        num_layers=1,
        dropout=0.0
    )
    model.eval()

    seq_len = 25
    batch_size = 2

    # Initial input tensor: [B, seq_len, 9]
    # Col 0: q_id, Col 1: part, Col 2: tag, Col 3-8: continuous (col 5 = is_correct)
    inputs_1 = torch.zeros(batch_size, seq_len, 9)
    inputs_1[:, :, 0] = torch.randint(1, 900, (batch_size, seq_len))
    inputs_1[:, :, 1] = torch.randint(1, 8, (batch_size, seq_len))
    inputs_1[:, :, 2] = torch.randint(1, 90, (batch_size, seq_len))
    inputs_1[:, :, 3:] = torch.randn(batch_size, seq_len, 6)

    # Forward pass 1
    with torch.no_grad():
        out_1 = model(inputs_1)

    assert out_1.shape == (batch_size, seq_len), f"Shape mismatch: {out_1.shape} vs {(batch_size, seq_len)}"

    # Modify future inputs starting at split_step
    split_step = 12
    inputs_2 = inputs_1.clone()
    # Drastically alter all features at and after split_step
    inputs_2[:, split_step:, 0] = torch.randint(1, 900, (batch_size, seq_len - split_step))
    inputs_2[:, split_step:, 1] = torch.randint(1, 8, (batch_size, seq_len - split_step))
    inputs_2[:, split_step:, 2] = torch.randint(1, 90, (batch_size, seq_len - split_step))
    inputs_2[:, split_step:, 3:] = inputs_2[:, split_step:, 3:] + 50.0  # Massive perturbation

    # Forward pass 2
    with torch.no_grad():
        out_2 = model(inputs_2)

    # Past predictions (steps 0 to split_step - 1) MUST be numerically identical
    past_diff = (out_1[:, :split_step] - out_2[:, :split_step]).abs().max().item()
    print(f"Max absolute discrepancy on past time steps [0:{split_step}]: {past_diff:.8e}")
    assert past_diff < 1e-6, f"Causality violation! Past outputs changed when future inputs changed. Diff: {past_diff}"

    # Future predictions SHOULD differ due to the perturbation
    future_diff = (out_1[:, split_step:] - out_2[:, split_step:]).abs().max().item()
    print(f"Max absolute change on future time steps [{split_step}:{seq_len}]: {future_diff:.4f}")
    assert future_diff > 1e-3, "Future outputs did not change as expected after perturbation."

    print("--> PASS: Causal forward invariance test succeeded!")


def test_zero_future_gradient():
    torch.manual_seed(42)
    model = CNNLSTM(
        num_questions=1000,
        num_parts=8,
        num_tags=100,
        embed_dim=32,
        hidden_dim=32,
        kernel_size=5,  # Test larger kernel size as well
        num_layers=1,
        dropout=0.0
    )
    model.eval()

    seq_len = 20
    inputs = torch.zeros(1, seq_len, 9, requires_grad=False)
    inputs[:, :, 0] = torch.randint(1, 500, (1, seq_len))
    inputs[:, :, 1] = torch.randint(1, 8, (1, seq_len))
    inputs[:, :, 2] = torch.randint(1, 50, (1, seq_len))

    # Make continuous features require gradients to test backward flow
    cont = torch.randn(1, seq_len, 6, requires_grad=True)

    # Forward with grad
    q_e = model.q_emb(inputs[:, :, 0].long())
    part_e = model.part_emb(inputs[:, :, 1].long())
    tag_e = model.tag_emb(inputs[:, :, 2].long())
    cont_e = model.cont_proj(cont)

    x = torch.cat([q_e, part_e, tag_e, cont_e], dim=-1)
    x = torch.relu(model.input_proj(x))
    x_conv = x.transpose(1, 2)
    x_conv_padded = torch.nn.functional.pad(x_conv, (model.kernel_size - 1, 0))
    x_conv = model.conv1(x_conv_padded)
    x_conv = model.bn1(x_conv)
    x_conv = torch.relu(x_conv)
    x_lstm = x_conv.transpose(1, 2)
    lstm_out, _ = model.lstm(x_lstm)
    logits = model.out_fc(lstm_out).squeeze(-1)

    target_step = 8
    loss = logits[0, target_step]
    loss.backward()

    # Gradients for all steps > target_step MUST be exactly zero
    future_grads = cont.grad[0, target_step + 1:, :]
    max_future_grad = future_grads.abs().max().item() if future_grads.numel() > 0 else 0.0
    print(f"Max future gradient at steps > {target_step}: {max_future_grad:.8e}")
    assert max_future_grad == 0.0, f"Future gradient leakage detected! Max grad = {max_future_grad}"

    print("--> PASS: Zero future gradient causality test succeeded!")


if __name__ == "__main__":
    test_causal_convolution_invariance()
    test_zero_future_gradient()
    print("\nALL CAUSAL CNN+LSTM VERIFICATION TESTS PASSED SUCCESSFULLY!")
