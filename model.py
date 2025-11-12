"""Nanochat model implementation and inference utilities (Multi-GPU support)."""

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
    
    def __init__(self, **kwargs):
        # Define expected fields with defaults
        expected_fields = {
            'sequence_len': 2048,
            'vocab_size': 65536,
            'n_layer': 32,
            'n_head': 16,
            'n_kv_head': 16,
            'n_embd': 2048,
        }
        
        # Set attributes from expected fields
        for field, default in expected_fields.items():
            setattr(self, field, kwargs.get(field, default))
        
        # Warn about unexpected parameters (optional)
        unexpected = set(kwargs.keys()) - set(expected_fields.keys())
        if unexpected:
            print(f"Warning: Ignoring unexpected config parameters: {unexpected}")


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
# GPT with rotary (Multi-GPU forward)
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
    ) -> torch.Tensor:
        """Forward pass supporting multi-GPU execution."""
        _b, t = idx.size()
        assert self.cos.size(1) >= t
        t0 = 0 if kv_cache is None else kv_cache.get_pos()

        # Move rotary buffers to same device as input
        device = idx.device
        cos_sin = self.cos[:, t0:t0 + t].to(device), self.sin[:, t0:t0 + t].to(device)

        x = self.transformer.wte(idx)
        x = x.to(dtype=self.dtype)
        x = norm(x)

        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)

        x = norm(x)
        logits = self.lm_head(x)
        softcap = 15
        return softcap * torch.tanh(logits / softcap)


# --------------------
# Pipeline Parallel wrapper for layer-wise distribution
# --------------------

class PipelineParallelGPT(nn.Module):
    """Distributes transformer layers across multiple GPUs in a pipeline."""
    
    def __init__(self, model: GPT, device_ids: list[int]):
        super().__init__()
        self.model = model
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
        # Calculate layers per device
        total_layers = len(model.transformer.h)
        self.layers_per_device = total_layers // self.num_devices
        self.remainder = total_layers % self.num_devices
        
        # Distribute layers across devices
        print(f"\nDistributing {total_layers} layers across {self.num_devices} GPUs:")
        layer_idx = 0
        self.device_map = {}
        
        for device_idx, device_id in enumerate(device_ids):
            # Give extra layers to first devices if there's a remainder
            num_layers = self.layers_per_device + (1 if device_idx < self.remainder else 0)
            device = torch.device(f"cuda:{device_id}")
            
            layer_range = list(range(layer_idx, layer_idx + num_layers))
            print(f"  GPU {device_id}: Layers {layer_range[0]}-{layer_range[-1]} ({num_layers} layers)")
            
            for l_idx in layer_range:
                self.device_map[l_idx] = device
                model.transformer.h[l_idx].to(device)
            
            layer_idx += num_layers
        
        # Place embedding and lm_head on first and last device
        self.first_device = torch.device(f"cuda:{device_ids[0]}")
        self.last_device = torch.device(f"cuda:{device_ids[-1]}")
        
        print(f"  Embedding on GPU {device_ids[0]}")
        print(f"  LM head on GPU {device_ids[-1]}")
        
        model.transformer.wte.to(self.first_device)
        model.lm_head.to(self.last_device)
        
        # Place rotary embeddings on first device
        model.cos = model.cos.to(self.first_device)
        model.sin = model.sin.to(self.first_device)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: object | None = None,
    ) -> torch.Tensor:
        """Forward pass with pipeline parallelism."""
        _b, t = idx.size()
        assert self.model.cos.size(1) >= t
        t0 = 0 if kv_cache is None else kv_cache.get_pos()
        
        # Start on first device
        x = idx.to(self.first_device)
        
        # Get rotary embeddings (already on first device)
        cos_sin = (
            self.model.cos[:, t0:t0 + t],
            self.model.sin[:, t0:t0 + t]
        )
        
        # Embedding
        x = self.model.transformer.wte(x)
        x = x.to(dtype=self.model.dtype)
        x = norm(x)
        
        # Process through layers, moving between devices as needed
        current_device = self.first_device
        for layer_idx, block in enumerate(self.model.transformer.h):
            target_device = self.device_map[layer_idx]
            
            # Move to new device if needed
            if target_device != current_device:
                x = x.to(target_device)
                # Move rotary embeddings to target device
                cos_sin = (cos_sin[0].to(target_device), cos_sin[1].to(target_device))
                current_device = target_device
            
            x = block(x, cos_sin, kv_cache)
        
        # Move to last device for final processing
        if current_device != self.last_device:
            x = x.to(self.last_device)
        
        x = norm(x)
        logits = self.model.lm_head(x)
        softcap = 15
        return softcap * torch.tanh(logits / softcap)


# --------------------
# Inference wrapper
# --------------------

class NanochatModel:
    """Loads weights and runs inference with multi-GPU support."""

    def __init__(
        self, 
        model_dir: str, 
        device: str = "cuda", 
        offload: bool | None = False,
        num_gpus: int | None = None
    ) -> None:
        """
        Args:
            model_dir: Directory containing model files.
            device: Device to use ("cuda" for multi-GPU).
            offload: Ignored; always False in GPU mode.
            num_gpus: Number of GPUs to use. If None, uses all available GPUs.
        """
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but GPU mode was requested.")
        
        # Determine number of GPUs
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            self.num_gpus = available_gpus
        else:
            self.num_gpus = min(num_gpus, available_gpus)
        
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available")
        
        print(f"Using {self.num_gpus} GPU(s) out of {available_gpus} available")
        
        # Primary device (GPU 0)
        self.device = torch.device("cuda:0")
        
        # List of all GPU devices to use
        self.device_ids = list(range(self.num_gpus))
        
        # hard-disable offload in GPU mode
        self.offload = False

        # perf knobs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        self.model_dir = model_dir
        # Load model with multi-GPU support
        self.model, self.dtype = self._load_model()
        self.enc = self._load_tokenizer()
        self._setup_special_tokens()

    # ---- weight / tokenizer loading ----

    def _load_model(self) -> tuple[nn.Module, torch.dtype]:
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

        # Load a sample weight to detect original dtype (load to CPU to avoid OOM)
        print(f"Detecting model dtype from {model_file}...")
        sample_weights = torch.load(model_file, map_location="cpu", weights_only=True)
        first_weight = next(iter(sample_weights.values()))
        original_dtype = first_weight.dtype
        print(f"Model original dtype: {original_dtype}")
        del sample_weights  # Free memory
        
        # Use original dtype
        dtype = original_dtype

        # Build empty model on META (no memory allocated)
        print("Building model architecture on meta device...")
        with torch.device("meta"):
            model = GPT(model_config, dtype=dtype)

        # Load weights directly to CPU first
        print(f"Loading weights from disk...")
        weights = torch.load(model_file, map_location="cpu", weights_only=True)
        weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}

        # If single GPU, load normally
        if self.num_gpus == 1:
            print(f"Loading to single GPU (cuda:0)...")
            model.to_empty(device=self.device)
            model.init_weights()
            model.load_state_dict(weights, strict=True, assign=True)
            model.to(self.device, dtype=dtype, non_blocking=True)
            model.cos = model.cos.to(self.device, non_blocking=True)
            model.sin = model.sin.to(self.device, non_blocking=True)
            model.eval()
            print("Model loaded successfully on single GPU")
            return model, dtype
        
        # Multi-GPU: Load weights distributed across devices
        print(f"Distributing model across {self.num_gpus} GPUs...")
        
        # Materialize model on CPU first
        model.to_empty(device="cpu")
        model.init_weights()
        model.load_state_dict(weights, strict=True, assign=True)
        
        # Wrap with pipeline parallel (this moves layers to different GPUs)
        model = PipelineParallelGPT(model, self.device_ids)
        model.eval()
        
        print("Model loaded successfully across multiple GPUs")
        return model, dtype

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

    # ---- generation (Multi-GPU) ----

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

        # create input on primary GPU
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Run in native model precision (no autocast)
        with torch.inference_mode():
            for _ in range(max_tokens):
                logits = self.model(x, kv_cache=None)
                logits = logits[:, -1, :]
                
                # Convert to float32 only for sampling stability
                logits = logits.float()
                
                # Apply temperature
                logits = logits / max(temperature, 1e-5)

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -1e10

                # Compute probabilities in fp32
                probs = F.softmax(logits, dim=-1)
                
                # Sample token
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = int(next_token.item())
                
                if token_id in self.stop_tokens:
                    break

                token_str = self.enc.decode([token_id])
                yield token_str

                # Move next_token to same device as x before concatenating
                next_token = next_token.to(x.device)
                x = torch.cat([x, next_token], dim=1)
