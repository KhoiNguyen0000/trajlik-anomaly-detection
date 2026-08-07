import torch
from src.denoiser import get_denoiser

denoiser = get_denoiser(
    model_type='dit', num_classes=15, input_shape=(272, 16, 16),
    z_channels=768, depth=8, width=1024,
    num_sampling_steps='3', patch_size=1
)

x = torch.randn(2, 272, 16, 16)
t = torch.zeros(2, dtype=torch.long)
final, z_seq, eps_seq, dz_seq = denoiser.ddim_reverse_sample(
    x, t, return_intermediates=True
)

assert len(z_seq) == 3, f'Expected 3 steps, got {len(z_seq)}'
assert len(dz_seq) == 3, f'Expected 3 deltas, got {len(dz_seq)}'
assert z_seq[0].shape == x.shape, f'Shape mismatch: {z_seq[0].shape}'
# Verify Δz_0 = z_τ1 - z_0
assert torch.allclose(dz_seq[0], z_seq[0] - x, atol=1e-5)
# Verify Δz_1 = z_τ2 - z_τ1
assert torch.allclose(dz_seq[1], z_seq[1] - z_seq[0], atol=1e-5)
print('passed: ddim_reverse_sample returns correct Δz')