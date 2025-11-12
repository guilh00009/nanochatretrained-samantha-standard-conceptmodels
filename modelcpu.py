"""Nanochat model implementation and inference utilities (GPU + CPU offload)."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Generator


# --------------------
# Config / helpers
# --------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    n_layer: int = 32
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 2048


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


_EXPECTED_NDIM = 4


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim == _EXPECTED_NDIM
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


# --------------------
# Model blocks
# --------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        kv_cache: object | None,
    ) -> torch.Tensor:
        b, t, _c = x.size()
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        tq = q.size(2)
        tk = k.size(2)
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)
        if kv_cache is None or tq == tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_mask = torch.zeros((tq, tk), dtype=torch.bool, device=q.device)
            prefix_len = tk - tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((tq, tq), dtype=torch.bool, device=q.device),
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        kv_cache: object | None,
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        return x + self.mlp(norm(x))


# --------------------
# GPT with rotary
# --------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig, *, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)],
                ),
            },
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # cast small, always-on modules eagerly
        self.transformer.wte.to(dtype=dtype)
        self.lm_head.to(dtype=dtype)

    def init_weights(self) -> None:
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 10000,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = "cpu"
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # keep trig buffers in model dtype for lower memory
        cos, sin = cos.to(self.dtype), sin.to(self.dtype)
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: object | None = None,
        *,
        offload: bool = False,
        run_device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Forward pass.

        When offload=True, each block is moved to `run_device` for its step and then
        returned to CPU to save VRAM.
        """
        _b, t = idx.size()
        assert self.cos.size(1) >= t
        t0 = 0 if kv_cache is None else kv_cache.get_pos()
        # ensure trig buffers are on the right device for the step
        if offload:
            cos_sin = (
                self.cos[:, t0:t0 + t].to(run_device, non_blocking=True),
                self.sin[:, t0:t0 + t].to(run_device, non_blocking=True),
            )
        else:
            cos_sin = self.cos[:, t0:t0 + t], self.sin[:, t0:t0 + t]

        x = self.transformer.wte(idx.to(self.transformer.wte.weight.device))
        x = x.to(dtype=self.dtype)
        x = norm(x)

        if offload:
            # stream each block onto GPU, compute, move back
            for block in self.transformer.h:
                block.to(run_device, dtype=self.dtype, non_blocking=True)
                x = x.to(run_device, non_blocking=True)
                x = block(x, cos_sin, kv_cache)
                x = x.to("cpu", non_blocking=True)
                block.to("cpu", dtype=self.dtype, non_blocking=True)
                torch.cuda.empty_cache()
        else:
            for block in self.transformer.h:
                x = block(x, cos_sin, kv_cache)

        x = norm(x)
        logits = self.lm_head(x.to(self.lm_head.weight.device))
        softcap = 15
        return softcap * torch.tanh(logits / softcap)


# --------------------
# Inference wrapper
# --------------------

class NanochatModel:
    """Loads weights and runs inference; prefers GPU with CPU offload fallback."""

    def __init__(self, model_dir: str, device: str = "auto", offload: bool | None = None) -> None:
        """
        Args:
            model_dir: Directory containing model files.
            device: "auto" (default), "cuda", or "cpu".
            offload: If True, keep blocks on CPU and stream them to GPU each step.
                     If None, auto-enable when using CUDA and total VRAM is limited (~<=6-8GB).
        """
        # Choose device and dtype
        use_cuda = torch.cuda.is_available() and (device in ("auto", "cuda"))
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        # sensible defaults for small-VRAM GPUs
        if offload is None:
            self.offload = self.device.type == "cuda"
        else:
            self.offload = bool(offload)

        # perf knobs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        self.model_dir = model_dir
        self.model = self._load_model()
        self.enc = self._load_tokenizer()
        self._setup_special_tokens()

    # ---- weight / tokenizer loading ----

    def _load_model(self) -> GPT:
        model_dir_path = Path(self.model_dir)
        model_files = list(model_dir_path.glob("model_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.model_dir}")
        model_file = model_files[0]

        meta_files = list(model_dir_path.glob("meta_*.json"))
        if not meta_files:
            raise FileNotFoundError(f"No meta files found in {self.model_dir}")
        meta_file = meta_files[0]

        with meta_file.open() as f:
            meta = json.load(f)

        model_config_kwargs = meta["model_config"]
        model_config = GPTConfig(**model_config_kwargs)

        # build empty model on CPU to control placement manually
        with torch.device("meta"):
            model = GPT(model_config, dtype=self.dtype)

        # load weights
        weights = torch.load(model_file, map_location="cpu", weights_only=True)
        weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}

        # cast to target dtype (fp16 on GPU; bf16 on CPU if available)
        desired = self.dtype
        weights = {k: (v.to(desired) if v.dtype != desired else v) for k, v in weights.items()}

        # materialize on CPU first (cheaper peak VRAM)
        model.to_empty(device="cpu")
        model.init_weights()
        missing, unexpected = model.load_state_dict(weights, strict=True, assign=True)
        assert not missing and not unexpected
        model.eval()

        if not self.offload and self.device.type == "cuda":
            # put full model on GPU if we are NOT offloading
            model.to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            # offload mode: keep blocks on CPU; small bits to CPU already
            model.to("cpu", dtype=self.dtype)

            # optional: keep embedding & head on GPU to save transfers
            if self.device.type == "cuda":
                model.transformer.wte.to(self.device, dtype=self.dtype, non_blocking=True)
                model.lm_head.to(self.device, dtype=self.dtype, non_blocking=True)

        # keep rotary buffers on CPU; we'll move slices to GPU per step when offloading
        model.cos = model.cos.to("cpu")
        model.sin = model.sin.to("cpu")
        return model

    def _load_tokenizer(self) -> object:
        tokenizer_path = Path(self.model_dir) / "tokenizer.pkl"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        with tokenizer_path.open("rb") as f:
            return pickle.load(f)

    def _setup_special_tokens(self) -> None:
        try:
            try:
                self.bos_token_id = self.enc.encode_single_token("<|bos|>")
            except KeyError:
                self.bos_token_id = self.enc.encode_single_token("<|endoftext|>")
            self.user_start_id = self.enc.encode_single_token("<|user_start|>")
            self.user_end_id = self.enc.encode_single_token("<|user_end|>")
            self.assistant_start_id = self.enc.encode_single_token("<|assistant_start|>")
            self.assistant_end_id = self.enc.encode_single_token("<|assistant_end|>")
            self.stop_tokens = {self.bos_token_id, self.assistant_end_id}
        except KeyError as e:
            raise ValueError(f"Required special token missing from tokenizer: {e}") from e

    # ---- formatting ----

    def format_prompt(self, message: str) -> list[int]:
        prompt_tokens = self.enc.encode_ordinary(message)
        return [
            self.bos_token_id,
            self.user_start_id,
            *prompt_tokens,
            self.user_end_id,
            self.assistant_start_id,
        ]

    def format_conversation(self, history: list[dict[str, str]]) -> list[int]:
        tokens = [self.bos_token_id]
        for message in history:
            role = message.get("role")
            content = message.get("content", "")
            content_tokens = self.enc.encode_ordinary(content)
            if role == "user":
                tokens.extend([self.user_start_id, *content_tokens, self.user_end_id])
            elif role == "assistant":
                tokens.extend([self.assistant_start_id, *content_tokens, self.assistant_end_id])
        tokens.append(self.assistant_start_id)
        return tokens

    # ---- generation ----

    def generate(
        self,
        prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Generator[str, None, None]:
        if history is not None:
            input_ids = self.format_conversation(history)
        elif prompt is not None:
            input_ids = self.format_prompt(prompt)
        else:
            raise ValueError("Either prompt or history must be provided")

        # start the context on CPU to reduce peak VRAM; move to GPU only when needed
        x = torch.tensor([input_ids], dtype=torch.long, device="cpu")

        use_cuda = self.device.type == "cuda"
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=self.dtype) if use_cuda else torch.cpu.amp.autocast(dtype=self.dtype)
        )

        with torch.inference_mode(), autocast_ctx:
            for _ in range(max_tokens):
                if self.offload and use_cuda:
                    # idx lives on CPU until we run; model.forward will stream blocks
                    logits = self.model(
                        x, kv_cache=None, offload=True, run_device=self.device
                    )
                else:
                    # keep everything on one device
                    x = x.to(self.device, non_blocking=True)
                    logits = self.model(x, kv_cache=None, offload=False, run_device=self.device)

                logits = logits[:, -1, :]
                logits = logits / max(temperature, 1e-5)

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if int(next_token.item()) in self.stop_tokens:
                    break

                token_str = self.enc.decode([int(next_token.item())])
                yield token_str

                # keep the running sequence on CPU in offload mode to avoid VRAM growth
                if self.offload and use_cuda:
                    x = torch.cat([x, next_token.to("cpu")], dim=1)
                    torch.cuda.empty_cache()
                else:
                    x = torch.cat([x, next_token], dim=1)
