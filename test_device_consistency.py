#!/usr/bin/env python3
"""Simple test for GPU device consistency"""

import torch
import sys

print("Testing tensor device consistency...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Test the problematic operation that was causing issues
target_q1 = torch.randn(32, 1, device=device)
target_q2 = torch.randn(32, 1, device=device)

# This was the problematic line - now without .cpu()
target_Q = torch.min(target_q1, target_q2)
print(f"target_Q device: {target_Q.device}")

# Test computation with other tensors on same device
current_q1 = torch.randn(32, 1, device=device)
current_q2 = torch.randn(32, 1, device=device)

# This should now work without device mismatch
try:
    loss = torch.nn.functional.mse_loss(current_q1, target_Q) + torch.nn.functional.mse_loss(current_q2, target_Q)
    print(f"Loss computed successfully: {loss.item():.4f}")
    print("✓ Device consistency test passed!")
except Exception as e:
    print(f"❌ Device consistency test failed: {e}")
    sys.exit(1)
