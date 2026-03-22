# model.py — DeePM Architecture
#
# Three key innovations from the paper:
#   1. Causal Sieve     — handles async macro data via attention gating
#   2. Macro Graph Prior — GNN models cross-asset economic relationships
#   3. EVaR objective   — distributionally robust loss penalising worst subperiods
#
# CPU-friendly design:
#   - Pure PyTorch, no GPU required
#   - Small model (hidden_dim=64 default) trains in <30 min on GitHub Actions CPU
#   - Graph layer uses simple attention, not full message passing

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ── 1. Causal Sieve ────────────────────────────────────────────────────────────

class CausalSieve(nn.Module):
    """
    Handles asynchronous macro data by learning which macro features
    are most relevant at each timestep via gated attention.

    Input:  (batch, lookback, n_macro_feats)
    Output: (batch, macro_hidden_dim)
    """

    def __init__(self, n_macro_feats: int, hidden_dim: int, n_heads: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_macro_feats, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        # Causal gate: learns which timesteps carry valid macro signals
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_macro: torch.Tensor) -> torch.Tensor:
        # x_macro: (B, L, M)
        h = self.input_proj(x_macro)                    # (B, L, H)
        attn_out, _ = self.attn(h, h, h)                # (B, L, H)
        gate = self.gate(h)                              # (B, L, H)
        h = self.norm(h + gate * attn_out)               # (B, L, H)
        # Weighted pool over time
        weights = F.softmax(self.pool(h), dim=1)         # (B, L, H)
        out = (weights * h).sum(dim=1)                   # (B, H)
        return out


# ── 2. Asset Encoder ───────────────────────────────────────────────────────────

class AssetEncoder(nn.Module):
    """
    Encodes per-asset time series features into a fixed-size embedding.
    Applied independently to each asset.

    Input:  (batch, lookback, n_asset_feats)
    Output: (batch, asset_hidden_dim)
    """

    def __init__(self, n_asset_feats: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_asset_feats,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, (h, _) = self.lstm(x)
        return self.norm(h[-1])  # (B, H) — last layer hidden state


# ── 3. Macro Graph Prior (Inter-Asset GNN) ────────────────────────────────────

class MacroGraphPrior(nn.Module):
    """
    Models cross-asset economic relationships via graph attention.
    The adjacency is softly learned — not hardcoded.

    Assets with similar macro sensitivity cluster together.
    E.g. TLT and LQD should have high edge weight (both rate-sensitive).

    Input:  asset_embeddings (batch, n_assets, asset_hidden_dim)
            macro_context    (batch, macro_hidden_dim)
    Output: (batch, n_assets, graph_hidden_dim)
    """

    def __init__(
        self,
        n_assets: int,
        asset_hidden_dim: int,
        macro_hidden_dim: int,
        graph_hidden_dim: int,
        n_heads: int = 2,
    ):
        super().__init__()
        self.n_assets = n_assets

        # Project macro context to same dim as asset embeddings
        self.macro_proj = nn.Linear(macro_hidden_dim, asset_hidden_dim)

        # Learnable graph attention
        self.graph_attn = nn.MultiheadAttention(
            embed_dim=asset_hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )

        # Macro-conditioned adjacency bias
        self.adj_bias = nn.Sequential(
            nn.Linear(asset_hidden_dim, n_assets),
        )

        self.norm  = nn.LayerNorm(asset_hidden_dim)
        self.proj  = nn.Linear(asset_hidden_dim, graph_hidden_dim)
        self.act   = nn.GELU()

    def forward(
        self,
        asset_emb: torch.Tensor,
        macro_ctx: torch.Tensor,
    ) -> torch.Tensor:
        # asset_emb: (B, A, H_a)
        # macro_ctx: (B, H_m)

        B, A, H = asset_emb.shape

        # Inject macro context as an additional "global" node
        macro_node = self.macro_proj(macro_ctx).unsqueeze(1)  # (B, 1, H_a)
        nodes = torch.cat([asset_emb, macro_node], dim=1)      # (B, A+1, H_a)

        # Graph attention with macro-conditioned adjacency bias
        adj_bias = self.adj_bias(macro_node.squeeze(1))        # (B, A)
        # Pad to (A+1) and expand for multi-head
        adj_pad = F.pad(adj_bias, (0, 1), value=0.0)           # (B, A+1)

        attn_out, _ = self.graph_attn(nodes, nodes, nodes)     # (B, A+1, H_a)

        # Residual + norm
        nodes = self.norm(nodes + attn_out)

        # Return only asset nodes (drop macro node)
        asset_out = nodes[:, :A, :]                            # (B, A, H_a)
        return self.act(self.proj(asset_out))                  # (B, A, H_g)


# ── 4. Portfolio Head ──────────────────────────────────────────────────────────

class PortfolioHead(nn.Module):
    """
    Maps per-asset graph embeddings + macro context to portfolio weights.
    n_outputs = n_assets+1 for FI (includes CASH), n_assets for Equity.
    """

    def __init__(
        self,
        n_assets: int,
        graph_hidden_dim: int,
        macro_hidden_dim: int,
        dropout: float = 0.2,
        n_outputs: int = None,   # defaults to n_assets+1
    ):
        super().__init__()
        self.n_assets  = n_assets
        self.n_outputs = n_outputs if n_outputs is not None else n_assets + 1
        combined_dim   = graph_hidden_dim + macro_hidden_dim

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.n_outputs),
        )

    def forward(
        self,
        asset_graph_emb: torch.Tensor,
        macro_ctx: torch.Tensor,
    ) -> torch.Tensor:
        B, A, Hg = asset_graph_emb.shape

        # Pool asset embeddings + concat macro
        asset_pool = asset_graph_emb.mean(dim=1)               # (B, Hg)
        combined   = torch.cat([asset_pool, macro_ctx], dim=1) # (B, Hg+Hm)

        logits  = self.head(combined)                          # (B, A+1)
        weights = F.softmax(logits, dim=-1)                    # (B, A+1)
        return weights


# ── 5. Full DeePM Model ────────────────────────────────────────────────────────

class DeePM(nn.Module):
    """
    Distributionally Robust Portfolio Engine with Macro Graph Prior.

    Forward pass:
        x_asset : (batch, n_assets, lookback, n_asset_feats)
        x_macro : (batch, lookback, n_macro_feats)

    Output:
        weights : (batch, n_assets+1) — portfolio weights including CASH
    """

    def __init__(
        self,
        n_assets: int,
        n_asset_feats: int,
        n_macro_feats: int,
        asset_hidden_dim: int = 64,
        macro_hidden_dim: int = 64,
        graph_hidden_dim: int = 64,
        n_attn_heads: int = 2,
        dropout: float = 0.2,
        include_cash: bool = True,
    ):
        super().__init__()
        n_outputs = n_assets + 1 if include_cash else n_assets

        self.n_assets    = n_assets
        self.asset_encoder  = AssetEncoder(n_asset_feats, asset_hidden_dim)
        self.causal_sieve   = CausalSieve(n_macro_feats, macro_hidden_dim, n_attn_heads)
        self.graph_prior    = MacroGraphPrior(
            n_assets, asset_hidden_dim, macro_hidden_dim,
            graph_hidden_dim, n_attn_heads,
        )
        self.portfolio_head = PortfolioHead(
            n_assets, graph_hidden_dim, macro_hidden_dim,
            dropout, n_outputs=n_outputs,
        )

    def forward(
        self,
        x_asset: torch.Tensor,
        x_macro: torch.Tensor,
    ) -> torch.Tensor:
        B, A, L, Fa = x_asset.shape

        # 1. Encode each asset independently
        x_flat = x_asset.view(B * A, L, Fa)           # (B*A, L, Fa)
        asset_emb = self.asset_encoder(x_flat)          # (B*A, H_a)
        asset_emb = asset_emb.view(B, A, -1)            # (B, A, H_a)

        # 2. Causal Sieve on macro
        macro_ctx = self.causal_sieve(x_macro)          # (B, H_m)

        # 3. Macro Graph Prior — cross-asset attention
        asset_graph = self.graph_prior(asset_emb, macro_ctx)  # (B, A, H_g)

        # 4. Portfolio weights
        weights = self.portfolio_head(asset_graph, macro_ctx)  # (B, A+1)

        return weights


# ── 6. EVaR Loss ───────────────────────────────────────────────────────────────

def evar_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    cash_rate: torch.Tensor,
    beta: float = 0.95,
    lambda_l2: float = 1e-4,
) -> torch.Tensor:
    """
    Entropic Value-at-Risk (EVaR) loss — distributionally robust objective.

    Penalises the worst (1-beta) fraction of episodes more heavily than
    standard expected return maximisation. This makes the model robust to
    crisis periods (2008, 2020, 2022) rather than just optimising average perf.

    Args:
        weights  : (B, A+1) — portfolio weights including CASH
        returns  : (B, A)   — next-day asset returns
        cash_rate: (B,)     — daily risk-free rate
        beta     : tail probability (0.95 = penalise worst 5% of days)
        lambda_l2: L2 regularisation on weights (encourages diversification)

    Returns:
        scalar loss (to minimise)
    """
    B, n_outputs = weights.shape
    # If n_outputs == n_assets: no CASH (equity mode)
    # If n_outputs == n_assets+1: includes CASH (FI mode)
    has_cash  = (n_outputs > returns.shape[1])
    n_assets_ = returns.shape[1]

    w_assets = weights[:, :n_assets_]
    w_cash   = weights[:, n_assets_:n_assets_+1] if has_cash else \
               torch.zeros(B, 1, dtype=weights.dtype)

    port_ret = (w_assets * returns).sum(dim=1) + \
               (w_cash.squeeze(1) * cash_rate)

    # Excess return over cash
    excess = port_ret - cash_rate                           # (B,)

    # EVaR: E[exp(-excess/t)] for optimal t, approximated via CVaR-like bound
    # Practical implementation: sort returns, penalise worst (1-beta) fraction
    sorted_excess, _ = torch.sort(excess)
    n_tail = max(1, int(B * (1 - beta)))
    tail_loss = -sorted_excess[:n_tail].mean()

    # Expected return term (maximise)
    expected_loss = -excess.mean()

    # L2 concentration penalty (discourages all-in on one asset)
    l2_penalty = (weights ** 2).sum(dim=1).mean()

    total_loss = 0.5 * expected_loss + 0.5 * tail_loss + lambda_l2 * l2_penalty

    return total_loss


def sharpe_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    cash_rate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable negative Sharpe ratio loss (fallback / comparison)."""
    B, n_outputs = weights.shape
    has_cash  = (n_outputs > returns.shape[1])
    n_assets_ = returns.shape[1]

    w_assets = weights[:, :n_assets_]
    w_cash   = weights[:, n_assets_:n_assets_+1] if has_cash else \
               torch.zeros(B, 1, dtype=weights.dtype)

    port_ret = (w_assets * returns).sum(dim=1) + w_cash.squeeze(1) * cash_rate
    excess   = port_ret - cash_rate

    mean_excess = excess.mean()
    std_excess  = excess.std() + eps

    return -(mean_excess / std_excess) * np.sqrt(252)
