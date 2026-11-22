# TinyRecursiveModel
Experiments with Tiny Recursive Model

This repository is a fork of [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels). The original README follows the sections below.

## What this fork is for

This fork is a place to learn how the Tiny Recursive Model works, in public. It is not a new model and it does not claim new results. Samsung's paper and code are the source. This fork is the notebook of someone working through them from the outside.

### Why this model

TRM is a small network, about 7 million parameters, that solves hard puzzles by looping over its own answer. It scores 87% on Sudoku-Extreme, where a network of the same kind without the loop scores 0%. That makes it a good model to study for three reasons.

- **It is small enough to read.** The core of the model is about twelve lines. The whole training script is one file.
- **It is small enough to run.** One training run fits on a single GPU, and a short one fits on a laptop.
- **The idea is unusual.** Most progress today comes from making models larger. TRM gets its results from reusing one tiny network many times.

### What "learning in public" means here

The work is done in the open, including the parts that go wrong.

- **The steps are small and written down.** Each change is one commit with a plain explanation of what changed and why.
- **Mistakes stay in the record.** The first Mac training run failed with a NaN loss. The cause and the fix are described below and in the commit history, not hidden.
- **Numbers are measured, and labeled when they are not.** Speeds and memory use come from real runs on a named machine. Estimates are called estimates.
- **Limits are stated.** Each section says what has not been tested.

### What has been done so far

- **Read the code and the paper.** That covers how the recursion, the halting and the training loop work.
- **Ported the code to Apple Silicon.** Samsung's code needs an NVIDIA GPU. The `mac-port` branch runs on a Mac.
- **Found and fixed a float32 bug in the loss.** The port exposed it. It does not show up on Samsung's setup.
- **Ran a short Sudoku training run on a Mac.** It proves that training works. It is far too short to produce a useful model.

### What comes next

These are plans, not results.

- **Smaller Sudoku boards.** A 6×6 board could train on a Mac in a reasonable time, and it would show the whole training loop from start to finish.
- **A full Sudoku run.** The paper's run takes about 18 hours on one L40S GPU. The goal is to reproduce the 87% result on a rented GPU.
- **Trying the idea on other step-by-step problems.** The open question is which other tasks can be written as a grid of tokens that a looping model can refine.

## Running on a Mac

Samsung's code needs an NVIDIA GPU with CUDA. The `mac-port` branch of this fork also runs on Apple Silicon through MPS, which is PyTorch's backend for the Mac's GPU. It still runs on CUDA, and it falls back to the CPU when neither is present.

### What changed

- **Device choice.** `pretrain.py` picks CUDA when it is present, then MPS, then the CPU. The hard-coded CUDA calls now use that one choice.
- **fp32 forward pass.** The model's forward pass runs in bfloat16 on CUDA. On MPS that gave a NaN loss after a single step, so every other device uses float32.
- **fp32 loss.** The stablemax loss casts to float64 on CUDA. MPS has no float64, so the loss runs in float32 there.
- **A loss fix for float32.** In float32, a logit of exactly 1.0 gave the stablemax loss an infinite gradient, and the whole model turned NaN. This is rare, so it showed up only after 50 or more training steps. The loss now clamps the input of its unused branch, which keeps the gradient finite. Results are unchanged where the branch is used.
- **The optimizer.** `adam-atan2` needs a CUDA build. When it is not installed, the code uses `adam-atan2-pytorch`, a pure-Python package with the same update rule and weight decay.
- **Compilation.** `torch.compile` relies on Triton, which supports only CUDA. It now runs only on CUDA.
- **Console metrics.** Training prints its metrics every 50 steps, and the eval prints its result. Before this, a run without wandb showed no loss.

### Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-mac.txt
```

### Check that it runs

This trains six steps on a random Sudoku-shaped batch. It needs no dataset and takes under a minute.

```bash
.venv/bin/python smoke_test.py 64
```

### A short training run

Build the Sudoku dataset first, with the command in the Sudoku-Extreme section of the original README below. Then run:

```bash
WANDB_MODE=offline .venv/bin/python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" global_batch_size=256 epochs=100 eval_interval=100 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 +run_name=mac_smoke ema=True
```

This is the paper's Sudoku setup, cut down to 100 epochs at batch 256. That is 390 steps. Drop `WANDB_MODE=offline` to log the run to your wandb account.

The eval at the end runs over the whole test set, which has 422,786 puzzles and takes a long time on a Mac. The run measured below used a copy of the dataset whose test set was cut to the first 2,000 puzzles.

### What to expect

These numbers were measured on an Apple M2 Max with 64 GB of memory, with Python 3.12 and torch 2.15.0.

| Batch size | Seconds per step | Memory |
|---|---|---|
| 128 | 1.34 | 14.1 GB |
| 256 | about 2.7 | 26.6 GB |
| 768 | does not fit | swaps past 38 GB |

- **Memory.** float32 costs about 0.1 GB per example. The paper's batch of 768 does not fit in 64 GB.
- **Speed.** Training runs at about 90 examples per second.
- **The short run.** Training took about 18 minutes. The training loss fell from 2.55 to 2.20. The eval scored 10.4% of cells correct and solved no puzzles. A loss near 2.4 means the model is guessing, so this run only proves that training works. It does not produce a useful model.
- **A full run.** The paper's Sudoku run is 50 million examples. At this speed that is about 6.5 days of training, so a full run still belongs on a rented GPU.

### Limits

- **CUDA is not re-tested.** The edited code has not been run on a CUDA machine since the port.
- **ARC eval does not work on MPS.** `evaluators/arc.py` still casts to float64. Sudoku does not use it.
- **Results may differ from the paper.** The Mac runs in float32 where CUDA uses bfloat16 and float64. Only a long run can show whether that changes accuracy.

---

**Update: Due to many automatically generated and irrelevant issues submitted to this repo (that have been deleted now) and our limited capacity to properly maintain this repo, we have to temporaliy archive (make read-only) this and several other repos.**


# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Motivation

Tiny Recursion Model (TRM) is a recursive reasoning model that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a tiny 7M parameters neural network. The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap. Currently, there is too much focus on exploiting LLMs rather than devising and expanding new lines of direction. With recursive reasoning, it turns out that “less is more”: you don’t always need to crank up model size in order for a model to reason and solve hard problems. A tiny model pretrained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

This work came to be after I learned about the recent innovative Hierarchical Reasoning Model (HRM). I was amazed that an approach using small models could do so well on hard tasks like the ARC-AGI competition (reaching 40% accuracy when normally only Large Language Models could compete). But I kept thinking that it is too complicated, relying too much on biological arguments about the human brain, and that this recursive reasoning process could be greatly simplified and improved. Tiny Recursion Model (TRM) simplifies recursive reasoning to its core essence, which ultimately has nothing to do with the human brain, does not require any mathematical (fixed-point) theorem, nor any hierarchy.

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

### Requirements

Installation should take a few minutes. For the smallest experiments on Sudoku-Extreme (pretrain_mlp_t_sudoku), you need 1 GPU with enough memory. With 1 L40S (48Gb Ram), it takes around 18h to finish. In case that you run into issues due to library versions, here is the requirements with the exact versions used: [specific_requirements.txt](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/specific_requirements.txt).

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

Expected: Around 87% exact-accuracy (+- 2%)

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

Expected: Around 75% exact-accuracy (+- 2%)

*Runtime:* < 20 hours

### Maze-Hard (assuming 4 L40S GPUs):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours

Actually, you can run Maze-Hard with 1 L40S GPU by reducing the batch-size with no noticable loss in performance:

```bash
run_name="pretrain_att_maze30x30_1gpu"
python pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours


### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days


## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
