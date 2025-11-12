# filename: nanochat_server.py
# Gradio + FastAPI (OpenAI-compatible) + Cloudflared public URL
# Correções:
# - Suporte completo a CORS + OPTIONS (sem 405 nos /v1/*)
# - UI Gradio com histórico real (multi-turn) e streaming
# - Mantida compatibilidade de API /v1/chat/completions com stream SSE
# - Extras de DX e logs
# - Aceita OpenAI "tools" (tools, tool_choice, parallel_tool_calls) e mensagens role="tool"
#   sem executar ferramentas (apenas compat) e sem repassar NADA ao modelo.

from __future__ import annotations

import os
import time
import uuid
import subprocess
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==========================
# Modelo (importe o seu)
# ==========================
from model import NanochatModel  # sua classe NanochatModel com .generate(...)

# Diretório padrão para os pesos
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()

_model: Optional[NanochatModel] = None


# ============================================================================
# Helpers
# ============================================================================
def coerce_text(content: Any) -> str:
    """Converte conteúdo OpenAI-style (string | list[parts] | dict) para string simples."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI às vezes envia lista de partes: [{"type":"text","text":"..."}, ...]
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # tenta campos comuns
                if "text" in p and isinstance(p["text"], str):
                    parts.append(p["text"])
                elif "content" in p and isinstance(p["content"], str):
                    parts.append(p["content"])
                elif "content" in p and isinstance(p["content"], list):
                    parts.append(coerce_text(p["content"]))
        return "".join(parts)
    if isinstance(content, dict):
        # tenta extrair "text" ou "content"
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content:
            return coerce_text(content["content"])
        # fallback: stringifica
        return ""
    # fallback
    return str(content)


# ============================================================================
# Schemas OpenAI-compatíveis (com tools) — tolerantes a extra fields
# ============================================================================
class ToolFunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None

    model_config = {"extra": "allow"}


class Tool(BaseModel):
    type: Literal["function"]
    function: ToolFunctionDef

    model_config = {"extra": "allow"}


class ToolChoiceFunction(BaseModel):
    name: str
    model_config = {"extra": "allow"}


class ToolChoiceTyped(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunction
    model_config = {"extra": "allow"}


ToolChoice = Any  # tolera str | dict | None




class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # OpenAI envia arguments como string JSON
    model_config = {"extra": "allow"}


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolCallFunction
    model_config = {"extra": "allow"}


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # Conteúdo pode vir como string, lista de partes, dict, etc.
    content: Any = ""
    # Campos opcionais para compat:
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    model_config = {"extra": "allow"}  # ignora campos desconhecidos


class ChatCompletionRequest(BaseModel):
    model: str = "nanochat"
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    stream: bool = False
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    stop: list[str] | str | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    user: str | None = None

    # ---- Campos OpenAI de tools (aceitos, mas não usados) ----
    tools: Optional[list[Tool]] = None
    tool_choice: Any = None
    parallel_tool_calls: Optional[bool] = None

    # Campos comuns adicionais aceitos silenciosamente (não usados)
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    stream_options: Optional[dict] = None  # aceito e ignorado

    model_config = {"extra": "allow"}  # evita 422 com chaves novas


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str
    logprobs: None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: str | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    # Não emitimos tool_calls no streaming (mantido para compat opcional)
    tool_calls: Optional[list[ToolCall]] = None

    model_config = {"extra": "allow"}


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    system_fingerprint: str | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ============================================================================
# Carregamento do modelo
# ============================================================================
def ensure_local_model_dir() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Crie a pasta 'model' ao lado deste script e coloque os arquivos do modelo "
            "(ex: meta_*.json, model_*.pt, token_bytes.pt, tokenizer.pkl), "
            "ou defina a variável de ambiente MODEL_DIR."
        )
    if not any(MODEL_DIR.iterdir()):
        raise FileNotFoundError(
            f"Model directory está vazio: {MODEL_DIR}\n"
            "Coloque os arquivos do modelo nessa pasta."
        )


def load_model() -> None:
    global _model
    if _model is None:
        ensure_local_model_dir()
        _model = NanochatModel(model_dir=str(MODEL_DIR), device=os.environ.get("DEVICE", "cpu"))


load_model()


# ============================================================================
# FastAPI App + CORS
# ============================================================================
app = FastAPI(title="Nanochat OpenAI-Compatible API", version="1.0.0")

# CORS amplo para permitir OPTIONS / preflight sem 405
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # afrouxado; ajuste em produção
    allow_credentials=False,
    allow_methods=["*"],          # inclui OPTIONS, GET, POST...
    allow_headers=["*"],          # Authorization, Content-Type, etc.
    expose_headers=["*"],
    max_age=86400,
)

# Rota coringa de OPTIONS para qualquer caminho
@app.options("/{rest_of_path:path}")
async def any_options(rest_of_path: str, request: Request):
    return PlainTextResponse("", status_code=204)


@app.get("/v1")
async def api_root():
    return JSONResponse({
        "message": "Nanochat OpenAI-Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
        }
    })


# Suporte GET e POST para /models, /v1/models
@app.get("/v1/models")
@app.get("/models")
@app.post("/v1/models")
@app.post("/models")
async def list_models():
    now = int(time.time())
    return ModelListResponse(
        object="list",
        data=[ModelInfo(id="nanochat", object="model", created=now, owned_by="nanochat")]
    )


@app.get("/v1/models/{model_id}")
@app.get("/models/{model_id}")
async def retrieve_model(model_id: str):
    if model_id != "nanochat":
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return ModelInfo(
        id="nanochat",
        object="model",
        created=int(time.time()),
        owned_by="nanochat"
    )


# ============================================================================
# Chat Completions
# ============================================================================
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Aceita nomes comuns mapeando para nanochat (não valida estritamente)
    # Aceita 'tools', 'tool_choice', 'parallel_tool_calls' — IGNORADOS por design.
    # role="tool" também é aceito, mas totalmente descartado (modelo NÃO vê nada).

    # --- Monta conversa sem qualquer contexto de tool ---
    conversation: list[dict[str, str]] = []
    for msg in request.messages:
        if msg.role == "tool":
            continue  # ignorar completamente
        if msg.role in ("system", "user", "assistant"):
            conversation.append({"role": msg.role, "content": coerce_text(msg.content)})
        # papéis desconhecidos: ignora

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            stream_chat_completion(
                conversation=conversation,
                completion_id=completion_id,
                created=created,
                model="nanochat",
                temperature=request.temperature,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
            ),
            media_type="text/event-stream",
        )

    # Resposta não-streaming
    full_response = ""
    token_count = 0
    for token in _model.generate(
        history=conversation,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    ):
        full_response += token
        token_count += 1

    # Não contar mensagens de tool nas métricas
    prompt_tokens = sum(len(coerce_text(m.content).split()) for m in request.messages if m.role != "tool")

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model="nanochat",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=full_response),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=token_count,
            total_tokens=prompt_tokens + token_count,
        ),
    )


async def stream_chat_completion(
    conversation: list[dict[str, str]],
    completion_id: str,
    created: int,
    model: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
) -> Generator[str, Any, None]:
    # Chunk inicial
    initial_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream de conteúdo
    for token in _model.generate(
        history=conversation,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    ):
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=token),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Chunk final
    final_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# Gradio UI com histórico real (multi-turn) + streaming
# ============================================================================
def build_gradio_ui() -> gr.Blocks:
    with gr.Blocks(title="nanochat") as demo:
        gr.Markdown("# 🤖 nanochat")
        gr.Markdown("Chat com um modelo treinado rapidinho (experimento de pesquisa).")
        gr.Markdown("**Aviso:** é pesquisa — não confie cegamente nas respostas. 😉")
        gr.Markdown("---")
        gr.Markdown("### 🌐 OpenAI-Compatible API")
        gr.Markdown(
            "Base URL: `<sua-url-publica>/v1`\n\n"
            "**Endpoints:**\n"
            "- `POST /v1/chat/completions`\n"
            "- `GET|POST /v1/models`\n"
            "- `GET /v1/models/{model_id}`"
        )
        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversa",
                    height=480,
                    bubble_full_width=False,
                    avatar_images=(None, None),
                )
                user_msg = gr.Textbox(placeholder="Mande sua mensagem aqui...", lines=3, autofocus=True)
                with gr.Row():
                    send_btn = gr.Button("Enviar", variant="primary")
                    clear_btn = gr.Button("Limpar", variant="secondary")

            with gr.Column(scale=2):
                temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                top_k = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Top-k sampling")
                system_prompt = gr.Textbox(
                    label="System message (opcional)",
                    placeholder="e.g., Você é um assistente conciso que responde em markdown.",
                    lines=6,
                )
                gr.Markdown("Dica: o histórico da sessão é usado como contexto a cada nova resposta.")

        # ------ Lógica com histórico e streaming ------
        def add_user_message(message: str, chat_history: list[tuple[str, str]]):
            if not message.strip():
                return gr.update(value=""), chat_history
            chat_history = chat_history + [(message, "")]
            return gr.update(value=""), chat_history

        def bot_respond(
            chat_history: list[tuple[str, str]],
            temperature: float,
            top_k: int,
            system_prompt: str,
        ):
            # Constrói a conversa completa (system + pares anteriores + último user)
            conversation: list[dict[str, str]] = []
            sys = (system_prompt or "").strip()
            if sys:
                conversation.append({"role": "system", "content": sys})

            for u, a in chat_history[:-1]:
                conversation.append({"role": "user", "content": u})
                conversation.append({"role": "assistant", "content": a})

            last_user = chat_history[-1][0]
            conversation.append({"role": "user", "content": last_user})

            partial = ""
            for token in _model.generate(
                history=conversation,
                max_tokens=512,
                temperature=temperature,
                top_k=top_k,
            ):
                partial += token
                chat_history[-1] = (last_user, partial)
                yield chat_history

        send_btn.click(
            add_user_message,
            inputs=[user_msg, chatbot],
            outputs=[user_msg, chatbot],
        ).then(
            bot_respond,
            inputs=[chatbot, temperature, top_k, system_prompt],
            outputs=[chatbot],
        )

        user_msg.submit(
            add_user_message,
            inputs=[user_msg, chatbot],
            outputs=[user_msg, chatbot],
        ).then(
            bot_respond,
            inputs=[chatbot, temperature, top_k, system_prompt],
            outputs=[chatbot],
        )

        def clear_history():
            return []

        clear_btn.click(clear_history, None, [chatbot])

    return demo


demo = build_gradio_ui()

# Monta o Gradio em /ui
app = gr.mount_gradio_app(app, demo, path="/ui")


# Redirect da raiz -> /ui
@app.api_route("/", methods=["GET", "HEAD", "POST"])
async def root_redirect():
    # 303 See Other instrui o client a refazer a requisição como GET
    return RedirectResponse(url="/ui/")


# ============================================================================
# Cloudflared tunnel (URL pública)
# ============================================================================
def start_cloudflare_tunnel(port: int = 7860):
    """Inicia um túnel Cloudflare via 'cloudflared' e retorna (public_url, process)."""
    try:
        process = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        url_pattern = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')

        public_url = None
        if process.stdout:
            for line in process.stdout:
                print(line.rstrip())
                match = url_pattern.search(line)
                if match:
                    public_url = match.group(0)
                    break

        return public_url, process

    except FileNotFoundError:
        print("\n❌ 'cloudflared' não encontrado!")
        print("\n📥 Instale uma das opções:")
        print("   pip install cloudflared   # (ou pycloudflared)")
        print("   Ou baixe em: https://github.com/cloudflare/cloudflared/releases")
        return None, None
    except Exception as e:
        print(f"\n❌ Erro iniciando o túnel: {e}")
        return None, None


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    import threading

    print("\n" + "="*70)
    print("🚀 Iniciando Nanochat Server com Acesso Público")
    print("="*70)

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print("\n⏳ Aguardando servidor subir...")
    time.sleep(3)

    print("\n🌐 Iniciando Cloudflare Tunnel...")
    public_url, tunnel_process = start_cloudflare_tunnel(7860)

    if public_url:
        print("\n" + "="*70)
        print("✅ PUBLIC URLs PRONTAS!")
        print("="*70)
        print(f"\n🌐 Gradio UI (Web Interface):")
        print(f"   {public_url}/ui")
        print(f"   ou: {public_url}")
        print(f"\n🔌 OpenAI-Compatible API Base URL:")
        print(f"   {public_url}/v1")
        print(f"\n📡 API Endpoints:")
        print(f"   - Models: GET|POST {public_url}/v1/models")
        print(f"   - Model Info: GET {public_url}/v1/models/nanochat")
        print(f"   - Chat: POST {public_url}/v1/chat/completions")
        print("\n" + "="*70)
        print("\n💡 cURL Example:")
        print(f"""
curl -X OPTIONS {public_url}/v1/models -i
curl -X GET {public_url}/v1/models

# Enviando tools / tool_choice (serão ignorados, apenas compatibilidade)
curl -X POST {public_url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "nanochat",
    "messages": [
      {{"role":"system","content":"Você é sucinto."}},
      {{"role":"user","content":"diga oi"}},
      {{"role":"tool","tool_call_id":"call_123","content":"resultado fake de tool"}}
    ],
    "tools": [{{"type":"function","function":{{"name":"search","description":"...", "parameters":{{"type":"object"}}}}}}],
    "tool_choice": "auto",
    "parallel_tool_calls": true,
    "temperature": 0.7,
    "stream": false
  }}'
        """)
        print("\n📝 Python OpenAI Client Example:")
        print(f"""
from openai import OpenAI
client = OpenAI(base_url="{public_url}/v1", api_key="dummy")  # api_key dummy

print(client.models.list())

resp = client.chat.completions.create(
    model="nanochat",
    messages=[{{"role":"user","content":"Hello!"}}],
    tools=[{{"type":"function","function":{{"name":"noop","parameters":{{"type":"object"}}}}}}],
    tool_choice="none",
    temperature=0.7
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="nanochat",
    messages=[{{"role":"user","content":"Conte uma história curtinha"}}],
    stream=True
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
        """)
        print("\n" + "="*70)
        print("⚠️ URL é temporária (muda ao reiniciar).")
        print("Ctrl+C para encerrar.")
        print("="*70 + "\n")
    else:
        print("\n⚠️ Não foi possível criar o túnel público.")
        print("Rodando localmente em:")
        print("  - UI:  http://localhost:7860/ui")
        print("  - API: http://localhost:7860/v1\n")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\n\n🛑 Encerrando...")
        if 'tunnel_process' in locals() and tunnel_process:
            tunnel_process.terminate()
