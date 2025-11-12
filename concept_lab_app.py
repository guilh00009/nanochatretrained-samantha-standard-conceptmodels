from __future__ import annotations

import io
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.hooks import RemovableHandle

from model import NanochatModel

try:
    from wordfreq import top_n_list
except Exception:
    top_n_list = None

# Directories
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()
CONCEPT_DIR = Path(__file__).resolve().parent / "concept_maps"
CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_HEADERS = ["concept", "prompt_samples", "peak_layer", "peak_norm", "saved_at"]

# Starter vocabulary used for random probing (fallback if library unavailable)
FALLBACK_WORDS = [
    "bread",
    "forest",
    "galaxy",
    "algorithm",
    "violin",
    "storm",
    "nebula",
    "compass",
    "pixel",
    "harbor",
    "quantum",
    "ember",
    "glacier",
    "lantern",
    "harp",
    "canyon",
    "mirror",
    "signal",
    "circuit",
    "petal",
    "spectrum",
    "meadow",
    "thunder",
    "breeze",
    "orbit",
    "granite",
    "aurora",
    "canvas",
    "harvest",
    "cobalt",
    "shadow",
    "turbine",
    "lattice",
    "caramel",
    "voyage",
    "kernel",
    "sunrise",
    "marble",
    "bamboo",
    "plasma",
    "anvil",
    "crystal",
    "velvet",
    "antler",
    "tapestry",
    "rift",
    "harpoon",
    "mech",
    "serenade",
    "delta",
    "cipher",
    "flame",
    "anchor",
    "plume",
    "nectar",
    "lighthouse",
    "prism",
    "specter",
    "compass",
    "brook",
    "satellite",
    "garden",
    "whisper",
]

WORD_LIBRARY_SIZE = 5000


def build_word_pool() -> List[str]:
    if top_n_list is None:
        return FALLBACK_WORDS
    try:
        raw_words = top_n_list("en", WORD_LIBRARY_SIZE)
    except Exception as exc:
        print(f"[concept_lab] Failed to load word list from wordfreq: {exc}")
        return FALLBACK_WORDS

    cleaned: List[str] = []
    for word in raw_words:
        w = word.strip().lower()
        if not w:
            continue
        if not w.isalpha():
            w = re.sub(r"[^a-z]", "", w)
        if not w or len(w) < 3 or len(w) > 12:
            continue
        cleaned.append(w)

    return cleaned or FALLBACK_WORDS


WORD_POOL = build_word_pool()

SIMPLE_PROMPT_TEMPLATES = [
    "{word}",
    "Say the word {word} plainly.",
    "Simple sentence: The {word} sits on the table.",
    "Short request: Mention {word} in one clause.",
    "Observation: I noticed a {word} nearby.",
    "Spell it: {letters}",
]

MODEL_LOCK = threading.Lock()

_MODEL: Optional[NanochatModel] = None
_MAPPER: Optional["ConceptMapper"] = None
_INJECTOR: Optional["ConceptInjector"] = None


def sample_words(limit: int, seed: Optional[int]) -> List[str]:
    rng = random.Random(seed or time.time())
    limit = max(1, int(limit))
    pool = WORD_POOL or FALLBACK_WORDS
    if limit <= len(pool):
        return rng.sample(pool, limit)
    return [rng.choice(pool) for _ in range(limit)]


def parse_seed(text: str | None) -> Optional[int]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


@dataclass
class ConceptRecord:
    concept: str
    prompts: List[str]
    layer_vectors: Dict[int, torch.Tensor]
    norms: Dict[int, float]
    highlight_layer: int
    created_at: float

    @property
    def safe_name(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.concept.lower()).strip("_")
        if slug:
            return slug
        return f"concept_{abs(hash(self.concept))}"

    def to_payload(self) -> dict:
        return {
            "concept": self.concept,
            "prompts": self.prompts,
            "layer_vectors": {int(k): v.cpu() for k, v in self.layer_vectors.items()},
            "norms": {int(k): float(v) for k, v in self.norms.items()},
            "highlight_layer": int(self.highlight_layer),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ConceptRecord":
        layer_vectors = {
            int(k): payload["layer_vectors"][k].cpu() for k in payload["layer_vectors"]
        }
        norms = {int(k): float(v) for k, v in payload["norms"].items()}
        return cls(
            concept=payload["concept"],
            prompts=list(payload["prompts"]),
            layer_vectors=layer_vectors,
            norms=norms,
            highlight_layer=int(payload["highlight_layer"]),
            created_at=float(payload["created_at"]),
        )

    def table_row(self) -> List[str]:
        prompt_preview = " | ".join(self.prompts)
        peak_norm = self.norms.get(self.highlight_layer, 0.0)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at))
        return [
            self.concept,
            prompt_preview,
            str(self.highlight_layer),
            f"{peak_norm:.4f}",
            timestamp,
        ]


class ConceptRepository:
    """Loads and persists mapped concept records."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, ConceptRecord] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        for payload_file in self.root.glob("*.pt"):
            try:
                payload = torch.load(payload_file, map_location="cpu")
                record = ConceptRecord.from_payload(payload)
                self.records[record.concept] = record
            except Exception as exc:
                print(f"[concept_repo] Failed to load {payload_file}: {exc}")

    def upsert(self, record: ConceptRecord) -> None:
        with self._lock:
            payload = record.to_payload()
            torch.save(payload, self.root / f"{record.safe_name}.pt")
            self.records[record.concept] = record

    def get(self, concept: str | None) -> Optional[ConceptRecord]:
        if concept is None:
            return None
        return self.records.get(concept)

    def list_concepts(self) -> List[str]:
        return sorted(self.records.keys())

    def table_rows(self) -> List[List[str]]:
        rows: List[List[str]] = []
        for record in sorted(self.records.values(), key=lambda r: r.created_at, reverse=True):
            rows.append(record.table_row())
        return rows


class ModelBridge:
    """Thin helper that exposes the base transformer layers."""

    def __init__(self, nano_model: NanochatModel):
        self.nano_model = nano_model
        self.base_model = self._unwrap(nano_model.model)
        self.layer_modules: List[nn.Module] = list(self.base_model.transformer.h)
        self.layer_count = len(self.layer_modules)

    def _unwrap(self, model: nn.Module) -> nn.Module:
        current = model
        while hasattr(current, "model"):
            current = current.model  # unwrap pipeline wrappers
        return current

    def run_forward(self, token_ids: List[int]) -> None:
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.nano_model.device)
        with torch.inference_mode():
            _ = self.nano_model.model(input_tensor)


class ConceptMapper:
    """Captures layer activations for prompts and aggregates them."""

    def __init__(self, nano_model: NanochatModel):
        self.nano_model = nano_model
        self.bridge = ModelBridge(nano_model)
        self.templates = SIMPLE_PROMPT_TEMPLATES
        self._baseline_vectors: Optional[Dict[int, torch.Tensor]] = None
        self.baseline_prompt = "Hold your thoughts and stay neutral."

    def _capture(self, token_ids: List[int]) -> Dict[int, torch.Tensor]:
        captured: Dict[int, torch.Tensor] = {}
        hooks = []

        for layer_idx, module in enumerate(self.bridge.layer_modules):
            def _hook(
                _module: nn.Module,
                _inputs: tuple,
                output: torch.Tensor,
                idx: int = layer_idx,
            ):
                if not isinstance(output, torch.Tensor):
                    return output
                captured[idx] = output[0, -1, :].detach().float().cpu()
                return output

            hooks.append(module.register_forward_hook(_hook))

        try:
            self.bridge.run_forward(token_ids)
        finally:
            for handle in hooks:
                handle.remove()

        return captured

    def _ensure_baseline(self) -> Dict[int, torch.Tensor]:
        if self._baseline_vectors is None:
            neutral_tokens = self.nano_model.format_prompt(self.baseline_prompt)
            self._baseline_vectors = self._capture(neutral_tokens)
        return self._baseline_vectors

    def _build_prompts(self, word: str, limit: int) -> List[str]:
        clean = " ".join(word.strip().split())
        if not clean:
            raise ValueError("Please provide a non-empty concept/word.")
        spelled = " ".join(list(clean.replace(" ", "")))
        prompts: List[str] = []
        cap = max(1, min(limit, len(self.templates)))
        for template in self.templates[:cap]:
            prompts.append(template.format(word=clean, letters=spelled))
        return prompts

    def map_concept(self, word: str, prompt_variations: int) -> ConceptRecord:
        prompts = self._build_prompts(word, prompt_variations)
        baseline = self._ensure_baseline()

        per_layer: Dict[int, List[torch.Tensor]] = {
            idx: [] for idx in range(self.bridge.layer_count)
        }

        for prompt in prompts:
            token_ids = self.nano_model.format_prompt(prompt)
            layer_outputs = self._capture(token_ids)
            for idx, vector in layer_outputs.items():
                delta = vector - baseline[idx]
                per_layer[idx].append(delta)

        layer_vectors: Dict[int, torch.Tensor] = {}
        norms: Dict[int, float] = {}
        for idx in range(self.bridge.layer_count):
            samples = per_layer[idx]
            if not samples:
                base = baseline[idx]
                layer_vectors[idx] = torch.zeros_like(base)
                norms[idx] = 0.0
                continue
            stacked = torch.stack(samples)
            avg = stacked.mean(dim=0)
            layer_vectors[idx] = avg
            norms[idx] = float(torch.norm(avg).item())

        highlight_layer = max(norms, key=norms.get) if norms else 0

        return ConceptRecord(
            concept=word.strip(),
            prompts=prompts,
            layer_vectors=layer_vectors,
            norms=norms,
            highlight_layer=int(highlight_layer),
            created_at=time.time(),
        )

    @staticmethod
    def render_concept_image(record: ConceptRecord) -> Optional[Image.Image]:
        if not record.norms:
            return None
        layers = sorted(record.norms.keys())
        values = np.array([[record.norms[idx] for idx in layers]], dtype=np.float32)
        max_val = float(values.max())
        if max_val > 0:
            values = values / max_val
        fig, ax = plt.subplots(figsize=(12, 2.6))
        im = ax.imshow(values, aspect="auto", cmap="magma")
        ax.set_yticks([])
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(idx) for idx in layers], rotation=90, fontsize=7)
        ax.set_xlabel("Transformer layer")
        ax.set_title(
            f"Concept '{record.concept}' activation map (peak layer {record.highlight_layer})"
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Relative activation", rotation=270, labelpad=12)
        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)
        plt.close(fig)
        buffer.seek(0)
        image = Image.open(buffer).copy()
        buffer.close()
        return image


class ConceptInjector:
    """Applies stored concept vectors during inference via forward hooks."""

    def __init__(self, bridge: ModelBridge):
        self.bridge = bridge
        self._handles: List[RemovableHandle] = []

    def enable(self, record: ConceptRecord, intensity: float = 1.0) -> None:
        intensity = max(0.0, float(intensity))
        if intensity <= 0.0:
            return
        self.disable()

        for layer_idx, vector in record.layer_vectors.items():
            module = self.bridge.layer_modules[layer_idx]
            delta = vector.clone().detach()

            def _hook(
                _module: nn.Module,
                _inputs: tuple,
                output: torch.Tensor,
                layer_delta: torch.Tensor = delta,
                level: float = intensity,
            ):
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    return output
                scaled = (layer_delta * level).to(output.device, dtype=output.dtype)
                output[:, -1, :] = output[:, -1, :] + scaled
                return output

            self._handles.append(module.register_forward_hook(_hook))

    def disable(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            handle.remove()


def get_model() -> NanochatModel:
    global _MODEL
    if _MODEL is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory {MODEL_DIR} not found. Set MODEL_DIR before running."
            )
        _MODEL = NanochatModel(str(MODEL_DIR))
    return _MODEL


def get_mapper() -> ConceptMapper:
    global _MAPPER
    if _MAPPER is None:
        _MAPPER = ConceptMapper(get_model())
    return _MAPPER


def get_injector() -> ConceptInjector:
    global _INJECTOR
    if _INJECTOR is None:
        _INJECTOR = ConceptInjector(get_mapper().bridge)
    return _INJECTOR


repository = ConceptRepository(CONCEPT_DIR)


def concept_table_rows() -> List[List[str]]:
    return repository.table_rows()


def dropdown_update(current_value: Optional[str]) -> gr.Dropdown:
    choices = repository.list_concepts()
    value = current_value if current_value in choices else None
    return gr.Dropdown.update(choices=choices, value=value)


def map_random_concepts(
    limit: int,
    prompt_variations: int,
    seed_text: str,
    current_concept: Optional[str],
) -> tuple[str, List[List[str]], Optional[Image.Image], gr.Dropdown]:
    seed = parse_seed(seed_text)
    words = sample_words(limit, seed)
    mapper = get_mapper()
    log_lines: List[str] = []
    latest_image: Optional[Image.Image] = None

    with MODEL_LOCK:
        for word in words:
            record = mapper.map_concept(word, prompt_variations)
            repository.upsert(record)
            latest_image = mapper.render_concept_image(record)
            peak = record.norms.get(record.highlight_layer, 0.0)
            log_lines.append(
                f"{record.concept}: peak layer {record.highlight_layer} (norm {peak:.4f})"
            )

    log_text = "\n".join(log_lines) if log_lines else "No concepts mapped yet."
    rows = repository.table_rows()
    dropdown = dropdown_update(current_concept)
    return log_text, rows, latest_image, dropdown


def map_single_concept(
    word: str,
    prompt_variations: int,
    current_concept: Optional[str],
) -> tuple[str, List[List[str]], Optional[Image.Image], gr.Dropdown]:
    clean = " ".join((word or "").strip().split())
    if not clean:
        return (
            "Please provide a word to map.",
            repository.table_rows(),
            None,
            dropdown_update(current_concept),
        )

    mapper = get_mapper()
    with MODEL_LOCK:
        record = mapper.map_concept(clean, prompt_variations)
        repository.upsert(record)
        image = mapper.render_concept_image(record)
    message = (
        f"Mapped concept '{record.concept}' "
        f"(peak layer {record.highlight_layer}, norm {record.norms.get(record.highlight_layer, 0.0):.4f})."
    )
    rows = repository.table_rows()
    dropdown = dropdown_update(current_concept)
    return message, rows, image, dropdown


def refresh_concepts(current_concept: Optional[str]) -> tuple[List[List[str]], gr.Dropdown]:
    return repository.table_rows(), dropdown_update(current_concept)


def add_user_message(message: str, history: List[tuple[str, str]]):
    if not message.strip():
        return gr.update(value=""), history
    updated = history + [(message, "")]
    return gr.update(value=""), updated


def concept_chat_reply(
    history: List[tuple[str, str]],
    temperature: float,
    top_k: int,
    system_prompt: str,
    selected_concept: Optional[str],
    inject_concept: bool,
    intensity: float,
) -> List[tuple[str, str]]:
    if not history:
        return history

    conversation: List[dict[str, str]] = []
    sys = (system_prompt or "").strip()
    if sys:
        conversation.append({"role": "system", "content": sys})

    for user_msg, bot_msg in history[:-1]:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": bot_msg})

    last_user = history[-1][0]
    conversation.append({"role": "user", "content": last_user})

    model = get_model()
    injector = get_injector()
    record = repository.get(selected_concept) if inject_concept else None

    response = ""
    try:
        with MODEL_LOCK:
            if record and intensity > 0:
                injector.enable(record, intensity)
            for token in model.generate(
                history=conversation,
                max_tokens=512,
                temperature=float(temperature),
                top_k=int(top_k),
            ):
                response += token
            history[-1] = (last_user, response)
    except Exception as exc:
        history[-1] = (last_user, f"[concept chat error] {exc}")
    finally:
        injector.disable()

    return history


def clear_history() -> List[tuple[str, str]]:
    return []


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Nanochat Concept Lab") as demo:
        gr.Markdown("# Nanochat Concept Lab")
        gr.Markdown(
            "Experimental space to probe the Nanochat model, map internal concept traces, "
            "and test how injecting those traces changes downstream answers."
        )

        with gr.Tab("Step 1 - Concept Mapper"):
            gr.Markdown(
                "Generate random probes, capture activations, and store reusable concept vectors."
            )
            with gr.Row():
                limit = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    label="Random concepts per sweep",
                )
                prompt_variations = gr.Slider(
                    minimum=1,
                    maximum=len(DEFAULT_PROMPT_TEMPLATES),
                    value=3,
                    step=1,
                    label="Prompt variations per concept",
                )
                seed_box = gr.Textbox(
                    label="Random seed (optional)",
                    placeholder="auto",
                )
            random_btn = gr.Button("Map random concepts", variant="primary")

            gr.Markdown("Map a specific word/concept and persist it for later injection.")
            with gr.Row():
                custom_word = gr.Textbox(
                    label="Word or short concept",
                    placeholder="e.g., bread",
                )
                custom_btn = gr.Button("Map this word", variant="secondary")

            log_box = gr.Textbox(label="Mapper log", lines=10)
            table = gr.Dataframe(
                headers=TABLE_HEADERS,
                value=concept_table_rows(),
                label="Saved concepts",
                interactive=False,
                wrap=True,
            )
            preview = gr.Image(label="Latest concept map", type="pil")
            refresh_btn = gr.Button("Refresh table", variant="secondary")

        with gr.Tab("Step 2 - Concept Chat"):
            gr.Markdown(
                "Chat with the model and optionally inject one of the mapped concepts into the residual stream."
            )
            chatbot = gr.Chatbot(
                label="Concept-aware chat",
                height=420,
                bubble_full_width=False,
            )
            user_box = gr.Textbox(
                label="Your message",
                placeholder="Ask something and optionally inject a concept...",
                lines=3,
                autofocus=True,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat", variant="secondary")

            with gr.Accordion("Generation controls", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.2,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=64,
                    step=1,
                    label="Top-k",
                )
                system_prompt = gr.Textbox(
                    label="System prompt (optional)",
                    placeholder="Set behavior or stay blank.",
                )

            with gr.Accordion("Concept injection", open=True):
                concept_selector = gr.Dropdown(
                    choices=repository.list_concepts(),
                    label="Mapped concept to inject",
                    value=None,
                )
                inject_toggle = gr.Checkbox(
                    label="Inject selected concept",
                    value=False,
                )
                intensity = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    value=1.0,
                    step=0.1,
                    label="Injection intensity",
                )

        # ----- wiring -----
        random_btn.click(
            map_random_concepts,
            inputs=[limit, prompt_variations, seed_box, concept_selector],
            outputs=[log_box, table, preview, concept_selector],
        )

        custom_btn.click(
            map_single_concept,
            inputs=[custom_word, prompt_variations, concept_selector],
            outputs=[log_box, table, preview, concept_selector],
        )

        refresh_btn.click(
            refresh_concepts,
            inputs=[concept_selector],
            outputs=[table, concept_selector],
        )

        user_box.submit(
            add_user_message,
            inputs=[user_box, chatbot],
            outputs=[user_box, chatbot],
        ).then(
            concept_chat_reply,
            inputs=[
                chatbot,
                temperature,
                top_k,
                system_prompt,
                concept_selector,
                inject_toggle,
                intensity,
            ],
            outputs=[chatbot],
        )

        send_btn.click(
            add_user_message,
            inputs=[user_box, chatbot],
            outputs=[user_box, chatbot],
        ).then(
            concept_chat_reply,
            inputs=[
                chatbot,
                temperature,
                top_k,
                system_prompt,
                concept_selector,
                inject_toggle,
                intensity,
            ],
            outputs=[chatbot],
        )

        clear_btn.click(clear_history, outputs=[chatbot])

    demo.queue(concurrency_count=1)
    return demo


demo = build_interface()


if __name__ == "__main__":
    print("Launching Nanochat Concept Lab on http://localhost:7861 ...")
    demo.launch(server_name="0.0.0.0", server_port=7861)
