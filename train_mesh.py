"""Train the mesh Sudoku model. See models/mesh_sudoku.py for the model itself.

This script is separate from pretrain.py on purpose. The mesh model has no halting,
no puzzle embeddings and no special optimizer, so it needs none of that machinery.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from models.mesh_sudoku import MeshSudoku, MeshTransformer

MODELS = {"sum": MeshSudoku, "attn": MeshTransformer}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(path: str):
    # The files store a blank as 1 and the digits 1 to 9 as 2 to 10.
    # The model wants a blank as 0 in its input, and the digits as classes 0 to 8 in its labels.
    x = np.load(f"{path}/all__inputs.npy").astype(np.uint8) - 1
    y = np.load(f"{path}/all__labels.npy").astype(np.uint8) - 2
    return x, y


def to_device(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.int64)).to(DEVICE)


@torch.no_grad()
def evaluate(model, x, y, passes, batch=1000):
    """Return the last-pass loss, the share of cells correct, and puzzles solved after each pass."""
    model.eval()
    loss = cells = 0.0
    solved = torch.zeros(passes, device=DEVICE)
    for i in range(0, len(x), batch):
        xb, yb = to_device(x[i:i + batch]), to_device(y[i:i + batch])
        logits = model(xb, passes)
        right = logits.argmax(-1) == yb
        loss += F.cross_entropy(logits[-1].flatten(0, 1), yb.flatten(), reduction="sum").item()
        cells += right[-1].sum().item()
        solved += right.all(-1).sum(-1)
    model.train()
    return loss / y.size, cells / y.size, (solved / len(x)).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sudoku-extreme-1k-aug-1000")
    p.add_argument("--name", default="mesh")
    p.add_argument("--model", choices=list(MODELS), default="sum", help="sum: plain neighbour sums. attn: attention along the mesh")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--passes", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-puzzles", type=int, default=2000)
    p.add_argument("--subset", type=int, default=0, help="train and eval on the first N training puzzles only, to test that the model can learn")
    p.add_argument("--full-eval", action="store_true", help="score the whole test set after the last step")
    p.add_argument("--resume", default="", help="checkpoint file to continue from")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    train_x, train_y = load(f"{args.data}/train")
    if args.subset:
        train_x, train_y = train_x[:args.subset], train_y[:args.subset]
        eval_x, eval_y = train_x, train_y
    else:
        test_x, test_y = load(f"{args.data}/test")
        pick = rng.choice(len(test_x), args.eval_puzzles, replace=False)
        eval_x, eval_y = test_x[pick], test_y[pick]

    model = MODELS[args.model](args.dim, args.passes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]

    out_dir = f"checkpoints/mesh/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(entity="shadowline", project="TRM", name=args.name, config=vars(args),
               mode="disabled" if args.no_wandb else "online")
    print(f"device {DEVICE}, weights {sum(p.numel() for p in model.parameters()):,}, "
          f"train puzzles {len(train_x):,}, eval puzzles {len(eval_x):,}", flush=True)

    t0 = time.time()
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)

        idx = rng.integers(0, len(train_x), args.batch)
        x, y = to_device(train_x[idx]), to_device(train_y[idx])
        logits = model(x)
        # Every pass is scored, not only the last. This teaches the model to improve the board at each step.
        loss = F.cross_entropy(logits.flatten(0, 2), y.expand(args.passes, -1, -1).flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # The same network runs many times in a row, so one bad step can blow up. The clip guards against that.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 50 == 0:
            sec = (time.time() - t0) / 50
            t0 = time.time()
            print(f"step {step} loss {loss.item():.4f} {sec:.3f} s/step", flush=True)
            wandb.log({"train/loss": loss.item(), "train/sec_per_step": sec}, step=step)

        if step % args.eval_every == 0 or step == args.steps:
            ev_loss, cells, solved = evaluate(model, eval_x, eval_y, args.passes)
            mem = torch.mps.driver_allocated_memory() / 2**30 if DEVICE == "mps" else 0.0
            print(f"eval step {step} loss {ev_loss:.4f} cells {cells:.4f} solved {solved[-1]:.4f} mem {mem:.1f} GB", flush=True)
            wandb.log({"eval/loss": ev_loss, "eval/cells": cells, "eval/solved": solved[-1]}, step=step)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)},
                       f"{out_dir}/last.pt")
            t0 = time.time()

    if args.full_eval and not args.subset:
        # Run twice as many passes as in training, to see whether extra passes keep helping.
        ev_loss, cells, solved = evaluate(model, test_x, test_y, 2 * args.passes)
        print(f"full test: puzzles {len(test_x):,} cells {cells:.4f} solved at pass {args.passes}: "
              f"{solved[args.passes - 1]:.4f}, at pass {2 * args.passes}: {solved[-1]:.4f}", flush=True)
        print("solved by pass: " + " ".join(f"{s:.3f}" for s in solved), flush=True)
        wandb.log({"test/cells": cells, "test/solved": solved[args.passes - 1], "test/solved_2x_passes": solved[-1]})
    wandb.finish()


if __name__ == "__main__":
    main()
