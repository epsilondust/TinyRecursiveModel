"""Train for a few steps on a random Sudoku-shaped batch.

This checks that the model, loss and optimizers run on this machine's device
without needing a dataset. Usage: python smoke_test.py [batch_size]
"""
import os
import sys
import time

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import pretrain
from dataset.common import PuzzleDatasetMetadata

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 64

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose("cfg_pretrain", overrides=[
        "arch.mlp_t=True", "arch.pos_encodings=none", f"global_batch_size={batch_size}", "evaluators=[]"])
config = pretrain.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore
metadata = PuzzleDatasetMetadata(
    pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=11, seq_len=81,
    num_puzzle_identifiers=1, total_groups=1000, mean_puzzle_examples=1, total_puzzles=1000, sets=["all"])
model, optimizers, optimizer_lrs = pretrain.create_model(config, metadata, rank=0, world_size=1)

device = pretrain.DEVICE
batch = {
    "inputs": torch.randint(1, 11, (batch_size, 81), dtype=torch.int32, device=device),
    "labels": torch.randint(2, 11, (batch_size, 81), dtype=torch.int32, device=device),
    "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32, device=device),
}
print(f"device={device} batch={batch_size} params={sum(p.numel() for p in model.parameters())}")

model.train()
with torch.device(device):
    carry = model.initial_carry(batch)  # type: ignore
for step in range(6):
    start = time.time()
    carry, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
    ((1 / batch_size) * loss).backward()
    for optim, lr in zip(optimizers, optimizer_lrs):
        for param_group in optim.param_groups:
            param_group["lr"] = lr
        optim.step()
        optim.zero_grad()
    loss_value = loss.item() / batch_size
    assert loss_value == loss_value, f"loss is NaN at step {step}"
    print(f"step {step}: loss per example {loss_value:.3f}, {time.time() - start:.2f}s")
print("OK")
