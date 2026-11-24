"""Mesh model for Sudoku.

The board is treated as a mesh. Each cell only hears from the cells it shares a rule with:
its row, its column and its box. One small network updates all 81 cells at once, and the
same network is applied many times, so the answer settles over the passes.
"""
import torch
import torch.nn.functional as F
from torch import nn


def neighbour_sums(m: torch.Tensor):
    """Sum the messages each cell receives from its row, its column and its box.

    m is [batch, 81, dim]. The cell's own message is left out of each sum.
    The sums are plain reductions over a 9 by 9 view, so no neighbour table is needed.
    """
    B, _, D = m.shape
    g = m.view(B, 9, 9, D)
    row = g.sum(2, keepdim=True).expand(-1, -1, 9, -1)
    col = g.sum(1, keepdim=True).expand(-1, 9, -1, -1)
    # View as (band, row in band, stack, column in stack), then sum inside each 3 by 3 box.
    box = g.view(B, 3, 3, 3, 3, D).sum((2, 4), keepdim=True).expand(-1, -1, 3, -1, 3, -1).reshape(B, 9, 9, D)
    return [(s - g).reshape(B, 81, D) for s in (row, col, box)]


class MeshSudoku(nn.Module):
    def __init__(self, dim: int = 96, passes: int = 32):
        super().__init__()
        self.passes = passes
        # Token 0 is a blank cell. Tokens 1 to 9 are the given digits.
        # There is no position embedding. A cell knows its neighbours and nothing else,
        # which makes row and column shuffles inside a band look identical to the model.
        self.embed = nn.Embedding(10, dim)
        self.message = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.update = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, 9)

    def forward(self, x: torch.Tensor, passes: int | None = None) -> torch.Tensor:
        """x is [batch, 81] with values 0 to 9. Returns digit scores [passes, batch, 81, 9]."""
        given = self.embed(x)
        h = given
        out = []
        for _ in range(passes or self.passes):
            sums = neighbour_sums(self.message(h))
            # The given digit is fed in on every pass so the clues cannot fade from the state.
            h = self.norm(h + self.update(torch.cat([h, given, *sums], -1)))
            out.append(self.readout(h))
        return torch.stack(out)


def relation_masks() -> torch.Tensor:
    """Return [3, 81, 81] booleans. masks[r, i, j] is true when cell j is a neighbour of cell i
    under relation r, where the relations are same row, same column and same box."""
    i = torch.arange(81)
    keys = (i // 9, i % 9, (i // 27) * 3 + i % 9 // 3)
    return torch.stack([(k[:, None] == k[None, :]) & (i[:, None] != i[None, :]) for k in keys])


class MeshTransformer(nn.Module):
    """The same loop and the same mesh as MeshSudoku, with attention in place of the plain sums.

    In MeshSudoku a cell sends one message, and every receiver gets the same one.
    Here the receiver's query meets each sender's key, so the receiver shapes what it hears.
    Each head attends along one relation only: a third of the heads see the row,
    a third see the column, and a third see the box.
    """

    def __init__(self, dim: int = 96, passes: int = 32, heads: int = 6):
        super().__init__()
        assert heads % 3 == 0 and dim % heads == 0
        self.passes, self.heads = passes, heads
        self.embed = nn.Embedding(10, dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, 9)
        # [1, heads, 81, 81]. Heads are grouped by relation: row heads, then column heads, then box heads.
        self.register_buffer("mask", relation_masks().repeat_interleave(heads // 3, 0)[None], persistent=False)

    def forward(self, x: torch.Tensor, passes: int | None = None) -> torch.Tensor:
        """x is [batch, 81] with values 0 to 9. Returns digit scores [passes, batch, 81, 9]."""
        B = x.shape[0]
        given = self.embed(x)
        h = given
        out = []
        for _ in range(passes or self.passes):
            # The given digit is added back on every pass so the clues cannot fade from the state.
            z = h + given
            q, k, v = self.qkv(z).view(B, 81, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)
            # The norm comes after each add. That keeps the state bounded however many passes run.
            h = self.norm1(z + self.proj(a.transpose(1, 2).reshape(B, 81, -1)))
            h = self.norm2(h + self.mlp(h))
            out.append(self.readout(h))
        return torch.stack(out)


if __name__ == "__main__":
    # Self-check: the reshaped sums must match a neighbour table built the slow way.
    torch.manual_seed(0)
    m = torch.randn(2, 81, 5, dtype=torch.float64)
    same = {
        "row": lambda i, j: i // 9 == j // 9,
        "col": lambda i, j: i % 9 == j % 9,
        "box": lambda i, j: (i // 27, i % 9 // 3) == (j // 27, j % 9 // 3),
    }
    for got, (name, rule) in zip(neighbour_sums(m), same.items()):
        table = torch.tensor([[float(i != j and rule(i, j)) for j in range(81)] for i in range(81)], dtype=torch.float64)
        assert (table.sum(1) == 8).all(), name
        assert torch.allclose(got, table @ m), name

    model = MeshSudoku()
    logits = model(torch.randint(0, 10, (4, 81)))
    assert logits.shape == (32, 4, 81, 9)
    print(f"ok: sums match the neighbour table. weights: {sum(p.numel() for p in model.parameters()):,}")

    # Self-check for the attention model: the masks hold 8 neighbours per relation and 20 in all,
    # and after one pass a cell changes only when a neighbour changes.
    masks = relation_masks()
    assert (masks.sum(2) == 8).all() and (masks.any(0).sum(1) == 20).all()
    model = MeshTransformer().eval()
    x = torch.randint(1, 10, (1, 81))
    base = model(x, 1)[0, 0]
    for j in range(81):
        y = x.clone()
        y[0, j] = (y[0, j] % 9) + 1
        changed = (model(y, 1)[0, 0] - base).abs().amax(-1) > 1e-6
        allowed = masks.any(0)[:, j].clone()
        allowed[j] = True
        assert not (changed & ~allowed).any(), j
        assert changed[allowed].all(), j
    assert model(x).shape == (32, 1, 81, 9)
    print(f"ok: attention stays on the mesh. weights: {sum(p.numel() for p in model.parameters()):,}")
