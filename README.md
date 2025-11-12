---
title: Nanochat
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
---

A lightweight chatbot,

Built with [Gradio](https://gradio.app) for the interface and [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index) for model distribution.

## Concept mapping lab

- `concept_lab_app.py` spins up a second Gradio interface focused on activation mapping and concept-aware chatting.
- Run it with `python concept_lab_app.py` (defaults to port `7861`).
- Step 1 lets you map random or custom words, persist the resulting layer vectors, and preview a PNG heatmap.
- Step 2 reuses those stored concepts so you can inject them during a normal chat session and observe the model's behavioral shifts.
- Random sweeps rely on [`wordfreq`](https://pypi.org/project/wordfreq/) to stream in simple, high-frequency vocabulary; a static fallback list is used only if that dependency is missing.

