"""Context-IRL library: reward model + IRL wrapper + training orchestrator.

Classes:
    RewardGPT / RewardGPTConfig — nanoGPT-style transformer with bidirectional
        attention and a per-token reward head (Linear -> sigmoid). Maps
        (B, N) token ids to (B, N) per-position rewards in (0, 1).
    ContextIRLReward — Context-IRL loss wrapper. Given a reward_net, samples
        K candidates per position (true + K-1 surrogates from tf or in_seq)
        and computes cross-entropy over discounted future rewards.
    ContextIRLOrchestrator — nanoGPT-style training loop. Takes the IRL
        wrapper, optimizer, scaler, ctx, data callables, and a config dict;
        run via .start().

Free functions (callable with explicit args, no module-level globals):
    get_batch(split, data_dir, block_size, batch_size, device, device_type)
    compute_tf_counts(data_dir, vocab_size)
    estimate_loss(irl, get_batch_fn, ctx, eval_iters)
    get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr)

Model code adapted from nanoGPT (https://github.com/karpathy/nanoGPT, MIT):
sites where Context-IRL diverges from upstream are marked with ``### MOD:``.
"""

import os
import math
import time
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# REWARD TRANSFORMER — nanoGPT model code, two marked modifications
# =============================================================================

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias.  (nanoGPT) """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """ nanoGPT's CausalSelfAttention.  ### MOD: is_causal=False (bidirectional for Context-IRL). """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head, self.n_embd, self.dropout = config.n_head, config.n_embd, config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,                       ### MOD: bidirectional
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)               ### MOD: no causal mask
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):                                  # nanoGPT
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):                                # nanoGPT
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class RewardGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer:    int = 2
    n_head:     int = 2
    n_embd:     int = 64
    dropout:  float = 0.0
    bias:      bool = False


class RewardGPT(nn.Module):
    """
    nanoGPT's GPT, modified at two sites:
      ### MOD 1: SelfAttention is bidirectional (see above)
      ### MOD 2: lm_head -> reward_head: Linear(n_embd, 1) + sigmoid, no weight tying.
    Contract: forward(idx: LongTensor (B, N)) -> Tensor (B, N) in (0, 1).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.reward_head = nn.Linear(config.n_embd, 1, bias=config.bias)   ### MOD 2

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"reward model parameters: {self.get_num_params()/1e6:.3f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return torch.sigmoid(self.reward_head(x)).squeeze(-1)              ### MOD 2

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # nanoGPT verbatim
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay   = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for _, p in param_dict.items() if p.dim() <  2]
        groups = [{'params': decay,   'weight_decay': weight_decay},
                  {'params': nodecay, 'weight_decay': 0.0}]
        fused_ok = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if (fused_ok and device_type == 'cuda') else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)


# =============================================================================
# CONTEXT-IRL WRAPPER
# =============================================================================

class ContextIRLReward(nn.Module):
    """
    Context-IRL reward learning over token sequences (LM-style, stateless).

    Wraps a transformer that maps (B, N) token ids -> (B, N) per-position rewards.
    """
    def __init__(self, reward_net, vocab_size, K=8, D=10, gamma=0.99,
                 surrogate_mode='uniform', tf_counts: torch.Tensor | None = None):
        super().__init__()
        self.reward_net = reward_net
        self.vocab_size = vocab_size
        self.K, self.D, self.gamma = K, D, gamma
        self.surrogate_mode = surrogate_mode
        if surrogate_mode == 'tf':
            assert tf_counts is not None and tf_counts.numel() == vocab_size
            probs = tf_counts.float() / tf_counts.float().sum()
            self.register_buffer('tf_probs', probs)

    def _surrogates_uniform(self, B, num, device):
        return torch.randint(0, self.vocab_size, (B, num), device=device)

    def _surrogates_tf(self, B, num, device):
        idx = torch.multinomial(self.tf_probs, B * num, replacement=True)
        return idx.view(B, num).to(device)

    def _surrogates_in_seq(self, tokens, pos, num):
        B, N = tokens.shape
        r = torch.randint(0, N - 1, (B, num), device=tokens.device)
        r = r + (r >= pos.unsqueeze(1)).long()
        return tokens.gather(1, r)

    def forward(self, tokens):
        """
        tokens: (B, N) LongTensor
        returns: (loss, acc) where acc is top-1 accuracy on the K-way ranking
        """
        B, N  = tokens.shape
        K, D, g = self.K, self.D, self.gamma
        device = tokens.device

        pos      = torch.randint(0, N - D + 1, (B,), device=device)            # (B,)
        true_tok = tokens.gather(1, pos.unsqueeze(1)).squeeze(1)               # (B,)
        if self.surrogate_mode == 'uniform':
            surr = self._surrogates_uniform(B, K - 1, device)
        elif self.surrogate_mode == 'tf':
            surr = self._surrogates_tf(B, K - 1, device)
        elif self.surrogate_mode == 'in_seq':
            surr = self._surrogates_in_seq(tokens, pos, K - 1)
        else:
            raise ValueError(self.surrogate_mode)
        cands = torch.cat([true_tok.unsqueeze(1), surr], dim=1)                # (B, K)

        expanded = tokens.unsqueeze(1).expand(B, K, N).clone()                 # (B, K, N)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        k_idx = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        p_idx = pos.unsqueeze(1).expand(B, K)
        expanded[b_idx, k_idx, p_idx] = cands

        r_pred = self.reward_net(expanded.view(B * K, N)).view(B, K, N)        # (B, K, N)

        j      = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        offset = j - pos.unsqueeze(1)
        valid  = (offset >= 0).to(r_pred.dtype)
        disc   = (g ** offset.clamp(min=0).to(r_pred.dtype)) * valid           # (B, N)
        Q      = (r_pred * disc.unsqueeze(1)).sum(dim=-1)                      # (B, K)

        target = torch.zeros(B, dtype=torch.long, device=device)
        loss   = F.cross_entropy(Q, target)
        with torch.no_grad():
            acc = (Q.argmax(dim=1) == target).float().mean()
        return loss, acc


# =============================================================================
# ORCHESTRATOR  (training loop, nanoGPT-style)
# =============================================================================

class ContextIRLOrchestrator:
    """Trains a Context-IRL reward model with a nanoGPT-style loop."""
    def __init__(
        self,
        irl,
        reward_net,
        reward_conf,
        optimizer,
        scaler,
        ctx,
        get_batch,
        estimate_loss,
        get_lr,
        config,
    ):
        self.irl = irl
        self.reward_net = reward_net
        self.reward_conf = reward_conf
        self.optimizer = optimizer
        self.scaler = scaler
        self.ctx = ctx
        self.get_batch = get_batch
        self.estimate_loss = estimate_loss
        self.get_lr = get_lr
        self.config = config

    def start(self):
        c = self.config
        iter_num = 0
        best_val_loss = 1e9
        X, _ = self.get_batch('train')
        t0 = time.time()
        print('-' * 10 + 'starting training' + '-' * 10)

        while True:
            lr = self.get_lr(iter_num) if c['decay_lr'] else c['learning_rate']
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            if iter_num % c['eval_interval'] == 0:
                m = self.estimate_loss()
                print(f"step {iter_num:>6d} | "
                      f"train loss {m['train_loss']:.4f} acc {m['train_acc']:.3f}  |  "
                      f"val loss {m['val_loss']:.4f} acc {m['val_acc']:.3f}  |  lr {lr:.2e}")
                if (m['val_loss'] < best_val_loss or c['always_save_checkpoint']) and iter_num > 0:
                    best_val_loss = min(best_val_loss, m['val_loss'])
                    ckpt = {
                        'reward_net':    self.reward_net.state_dict(),
                        'optimizer':     self.optimizer.state_dict(),
                        'reward_conf':   self.reward_conf.__dict__,
                        'iter_num':      iter_num,
                        'best_val_loss': best_val_loss,
                        'config':        c,
                    }
                    torch.save(ckpt, os.path.join(c['out_dir'], 'ckpt.pt'))
                    print(f"  saved checkpoint -> {c['out_dir']}/ckpt.pt")

            if iter_num == 0 and c['eval_only']:
                break

            with self.ctx:
                loss, acc = self.irl(X)
            X, _ = self.get_batch('train')           # async prefetch (nanoGPT pattern)
            self.scaler.scale(loss).backward()
            if c['grad_clip'] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), c['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time(); dt = t1 - t0; t0 = t1
            if iter_num % c['log_interval'] == 0:
                print(f"iter {iter_num:>6d}: loss {loss.item():.4f}  acc {acc.item():.3f}  "
                      f"lr {lr:.2e}  dt {dt*1000:.0f}ms")
            iter_num += 1
            if iter_num > c['max_iters']:
                break


# =============================================================================
# DATA / EVAL / LR HELPERS (free functions; explicit args, no globals)
# =============================================================================

def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    """nanoGPT-style memmap batch sampler. Returns (x, y) of shape (B, N)."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64))     for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def compute_tf_counts(data_dir, vocab_size):
    """Stream train.bin through np.bincount in chunks (works for OpenWebText too)."""
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    counts = np.zeros(vocab_size, dtype=np.int64)
    chunk = 10_000_000
    for i in range(0, len(data), chunk):
        counts += np.bincount(np.asarray(data[i:i+chunk], dtype=np.int64), minlength=vocab_size)
    counts = counts.astype(np.float32) + 1.0          # Laplace smoothing
    print(f"computed unigram counts over {len(data):,} tokens; vocab seen: {(counts > 1).sum()}")
    return torch.from_numpy(counts)


@torch.no_grad()
def estimate_loss(irl, get_batch_fn, ctx, eval_iters):
    """Average (loss, acc) over `eval_iters` batches on train and val splits."""
    out = {}
    irl.eval()
    for split in ('train', 'val'):
        losses, accs = torch.zeros(eval_iters), torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, _ = get_batch_fn(split)
            with ctx:
                loss, acc = irl(x)
            losses[i], accs[i] = loss.item(), acc.item()
        out[f'{split}_loss'] = losses.mean().item()
        out[f'{split}_acc']  = accs.mean().item()
    irl.train()
    return out


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """Linear warmup + cosine decay schedule from nanoGPT."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    r = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (learning_rate - min_lr)
