# hardware_qa_assistant_lang_graph_gradio.py — Advanced Hardware QA Assistant
# ---------------------------------------------------------------------------------
# What this script demonstrates:
#   • A real multi‑node LangGraph with conditional routing and checkpoints
#   • Multimodal ingestion (PDF/DOCX/Images → OCR) decoupled from Q&A
#   • Persistent, per‑workspace RAG (Chroma) with local Sentence‑Transformers embeddings
#   • Optional online search (Tavily) fused with RAG, with citations
#   • Answer verification (heuristic) + targeted clarification branch
#   • Explicit session termination that generates a Markdown report (downloadable)
#   • Provider abstraction using OpenAI‑compatible API (Together primary, Groq hook ready)
#
# Run steps:
#   1) pip install -r requirements.txt
#   2) Copy .env.example → .env and set at least:
#        OPENAI_API_KEY=your_together_key
#        OPENAI_BASE_URL=https://api.together.xyz/v1
#   3) python hardware_qa_assistant_lang_graph_gradio.py  → open the Gradio link
from __future__ import annotations

import os
import io
import json
import tempfile
from typing import TypedDict, List, Dict, Any, Tuple, Optional
import time
import re, urllib.parse
from jose import jwt

from dotenv import load_dotenv
load_dotenv()

# LangGraph core
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# HTTP + UI
import requests
# Groq SDK
from groq import Groq
import gradio as gr

import contextvars
_progress_var = contextvars.ContextVar("progress_hook", default=None)

# Retrieval pipeline knobs (NEW)
MQ_N = 10                # how many sub-queries to generate
POOL_K_EACH = 40         # top-k per sub-query from Chroma
FUSE_K = 60              # keep this many after RRF fusion (before rerank)
MMR_KEEP = 8             # final number of chunks passed to the LLM
MMR_LAMBDA = 0.7         # relevance vs diversity trade-off
PROMOTE_SIBLINGS = 1     # siblings per parent (controls token growth)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _set_progress_hook(h): _progress_var.set(h)
def _get_progress_hook(): return _progress_var.get()

from hardware_qa_assistant_helpers import (
    # RAG + workspace
    get_collection, add_docs, retrieve, retrieve_info, list_projects_for_user, delete_project,
    # Manage PDFs
    list_workspace_files_by_filename, delete_files_by_filename,
    # Parsers
    parse_image, parse_pdf_pages, parse_docx_sections, chunk_text,
    # Reports
    list_summary_reports, read_report_text,
    # Retrieval quality utils  ← add these
    rrf_fuse, rerank_cross_encoder, mmr_select, promote_parents,
    # Trace
    trace_push, render_trace_artifacts,
)

# --- Compact uploader CSS (Option 2) ---
UPLOADER_CSS = """
/* keep the dropzone narrow */
#uploader { max-width: 520px; }

/* trim vertical padding around the dropzone */
#uploader .gradio-container,
#uploader .container,
#uploader > div { padding-top: 0 !important; padding-bottom: 0 !important; }

/* clamp the drop area height (selectors vary by gradio version, include both) */
#uploader .h-full,
#uploader [data-testid="file"] { min-height: 80px !important; height: 80px !important; }

/* keep preview small but scrollable if many files */
#uploader .file-preview, 
#uploader .file-preview-box { max-height: 50px; overflow: auto; }

/* ===== User bubble text color (make visible on white bubble) ===== */
#chatui .message.user,
#chatui [data-testid="user"],
#chatui .chat-message.user,
#chatui .user {
  color: #0d47a1 !important;           /* dark blue text */
}

/* Make everything inside inherit the same color (stronger than theme rules) */
#chatui .message.user * { color: inherit !important; }

/* Optional: links slightly darker so they still look like links */
#chatui .message.user a { color: #0a3d91 !important; }
/* Ensure Markdown wrapper used by Gradio adopts the blue color, too */
#chatui .message.user .prose,
#chatui .message.user .prose :is(p, span, li, strong, em, h1, h2, h3, h4, h5, h6) {
  color: #0d47a1 !important;
}

/* Code blocks/inline code also visible on white bubble */
#chatui .message.user :is(code, pre) {
  color: #0d47a1 !important;
  background: rgba(13, 71, 161, 0.08) !important; /* subtle tint so code stands out */
  border-color: rgba(13, 71, 161, 0.18) !important;
}

/* Blockquotes inherit the blue and show a matching left border */
#chatui .message.user blockquote {
  color: #0d47a1 !important;
  border-left-color: #0d47a1 !important;
}
/* Chat tab: make the “User: user@example.com” line yellow (and its link) */
#user_label, 
#user_label a, 
#user_label .prose,
#user_label .prose a {
  color: #ffeb3b !important;   /* bright yellow */
  text-decoration-color: #ffeb3b !important;
}

/* Trace tab: make web citation links yellow so they’re readable */
#trace_md a,
#trace_md .prose a,
#trace_md a:visited,
#trace_md .prose a:visited {
  color: #ffeb3b !important;
  text-decoration-color: #ffeb3b !important;
}

/* compact the schematic checkbox */
#schem_tick, #schem_tick > div { padding-top: 0 !important; padding-bottom: 0 !important; }
#schem_tick * { margin-top: 0 !important; margin-bottom: 0 !important; }
#schem_tick label { font-size: 14px !important; line-height: 1.1 !important; }
#send_btn { margin-top: 8px !important; margin-bottom: 6px !important; }

"""

# Web search (optional)
try:
    from tavily import TavilyClient
except Exception:  # allow running without tavily installed
    TavilyClient = None  # type: ignore

# =====================
# Config (from .env; overridable via the UI)
# =====================
# OpenAI-compatible env (Groq only)
PROVIDER     = os.getenv("LLM_PROVIDER", "groq").lower()   # groq
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")
#MODEL_G      = os.getenv("LLM_MODEL_GROQ", "llama-3.1-70b-versatile")
MODEL_G      = os.getenv("LLM_MODEL_GROQ", "openai/gpt-oss-120b")
# Vision (Groq Scout, image understanding)
VISION_MODEL_T = os.getenv("VISION_MODEL_GROQ", "meta-llama/llama-4-scout-17b-16e-instruct")



OCR_BACKEND  = os.getenv("OCR_BACKEND", "paddle").lower()      # paddle | tesseract | auto
CHROMA_DIR   = os.getenv("CHROMA_DIR", ".chroma_hwqa_workspace")
DEFAULT_WS   = os.getenv("WORKSPACE", "default")
TAVILY_KEY   = os.getenv("TAVILY_API_KEY", "")
DEFAULT_LANGUAGE = "English"  # could be auto or Greek if desired
JWT_LEEWAY_SEC = int(os.getenv("JWT_LEEWAY_SEC", "900"))

import re
_URL_RE = re.compile(r"https?://[^\s)\]]+", flags=re.I)
_SOURCES_RE = re.compile(r"(?is)\n+(?:sources?|references?)\s*:.*$", flags=re.I)

def sanitize_answer(
    text: str,
    allowed_urls: List[str],
    *,
    append_refs: bool = True,
    title: str = "Citations (web)"
) -> str:
    """
    1) Remove hallucinated 'Sources/References' blocks.
    2) Strip any raw URLs not in allowed_urls.
    3) Optionally append a vetted citations list so links show in the Chat.
    """
    if not text:
        return text

    out = _SOURCES_RE.sub("", (text or "").strip())

    if allowed_urls:
        allow = tuple(allowed_urls)
        def _sub(m):
            u = m.group(0)
            return u if u.startswith(allow) else ""   # keep only vetted URLs
        out = _URL_RE.sub(_sub, out)

        if append_refs:
            refs = "\n".join(f"[{i+1}] {u}" for i, u in enumerate(allowed_urls, start=1))
            out = f"{out}\n\n**{title}:**\n{refs}"
    else:
        # No vetted links for this turn → remove any raw URLs the model emitted
        out = _URL_RE.sub("", out)

    return out.strip()

# Optional Tavily client (clean SERP API with free tier)
tavily = TavilyClient(api_key=TAVILY_KEY) if (TAVILY_KEY and TavilyClient) else None

# =====================
# Provider‑agnostic Chat wrapper via OpenAI‑compatible endpoints
# =====================

def chat_llm(messages: List[Dict[str, Any]], *, provider: str, model: str, max_tokens: int = 3000, temperature: float = 0.2) -> Tuple[str, Dict[str, Any]]:
    """Minimal wrapper around Groq using OpenAI‑compatible /chat/completions.
    - messages: OpenAI-style list[{role, content}]
    - provider: 'together' or 'groq'
    - model: model id for that provider
    """
    provider = provider.lower()
    # Groq OpenAI‑compatible endpoint
    if not GROQ_KEY:
        return "[LLM error] GROQ_API_KEY missing; switch provider or set key.", {}
    base = "https://api.groq.com/openai/v1"
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
        if r.status_code >= 400:
            return f"[LLM error] {r.status_code} {r.reason}: {r.text}", {}
        data = r.json()
        reply = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        return reply, usage
    except Exception as e:
        return f"[LLM error] {e}", {}

def call_vision_extract(image_bytes: bytes, *, model: Optional[str] = None) -> Tuple[Optional[dict], str, str, Optional[dict]]:
    """
    Call Groq 'meta-llama/llama-4-scout-17b-16e-instruct' for schematic understanding.
    Returns (json_obj, summary_text, raw_text, usage_dict). On error, json_obj=None and summary_text starts with [vision error].
    """
    api_key = os.getenv("GROQ_API_KEY") or GROQ_KEY
    if not api_key:
        return None, "[vision error] missing GROQ_API_KEY", "", {}

    import base64, re
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    # Keep your existing strict-JSON + 'summary:' prompt contract,
    # so downstream parsing remains unchanged.
    prompt = (
        "You are an EE expert. From the schematic IMAGE, output STRICT JSON first, "
        "then exactly one line starting with 'summary:' (1–3 sentences).\n"
        "JSON schema:\n"
        "{\n"
        '  "components":[{"ref":"U1","label":"PI6C48545LE","type":"IC","pins":["CLK0","CLK1","OE","VCC1","GND1"]}],\n'
        '  "nets":[{"name":"REFCLK_2G5","connections":[{"ref":"U1","pin":"Q0"},{"ref":"J1","pin":"P1"}]}],\n'
        '  "texts":["3.3V","100nF","OE","CLK_SEL"]\n'
        "}\n"
        "Rules: JSON ONLY as the first output (no prose). Then one 'summary:' line."
    )

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=(model or VISION_MODEL_T),  # now defaults to Groq Scout
            temperature=0.1,
            max_completion_tokens=1500,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        usage = {}
        try:
            # Groq returns usage fields; guard them defensively.
            u = getattr(completion, "usage", None)
            if u:
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None),
                }
        except Exception:
            usage = {}
    except Exception as e:
        return None, f"[vision error] {e}", "", {}

    # Parse JSON block + 'summary:' line (unchanged)
    j = None
    try:
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(text[start:end+1])
    except Exception:
        j = None

    m = re.search(r"summary\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    summary = (m.group(1).strip() if m else "") or text[:800]
    return j, summary, text, usage


# =====================
# User & workspace helpers (per-user collections)
# =====================
AUTH_SECRET = os.getenv("AUTH_SECRET", "change-me")
JWT_ALG = "HS256"

def _slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+','-', (s or '').strip().lower()).strip('-') or 'default'


def get_logged_in_email_from_request(request: gr.Request) -> Optional[str]:
    """
    Return logged-in email. First try the readable 'who' cookie (display only),
    then decode the HttpOnly 'session' JWT using AUTH_SECRET.
    Works across Gradio versions: request.cookies or Cookie header fallback.
    """
    # 1) Non-HttpOnly display cookie (simple path)
    try:
        if hasattr(request, "cookies") and isinstance(request.cookies, dict):
            who = request.cookies.get("who")
            if who:
                return who
    except Exception:
        pass

    # 2) HttpOnly JWT cookie
    tok = None
    try:
        if hasattr(request, "cookies") and isinstance(request.cookies, dict):
            tok = request.cookies.get("session")
    except Exception:
        tok = None

    # 3) Fallback: parse Cookie header
    if not tok:
        cookie_hdr = request.headers.get("cookie", "")
        m = re.search(r"(?:^|;)\s*session=([^;]+)", cookie_hdr, flags=re.IGNORECASE)
        tok = urllib.parse.unquote(m.group(1)) if m else None

    if not tok:
        return None

    try:
        import time  # ensure available
        # Verify signature, but skip built-in exp check
        payload = jwt.decode(
            tok,
            AUTH_SECRET,
            algorithms=[JWT_ALG],
            options={"verify_exp": False},  # jose here also lacks 'leeway' kw
        )
        # Manual leeway on exp
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            now = int(time.time())
            if now > int(exp) + JWT_LEEWAY_SEC:
                return None
        return payload.get("sub")
    except Exception:
        return None


# =====================
# State — explicit fields make the graph easier to reason about and demo
# =====================
class State(TypedDict, total=False):
    # Configuration & context
    thread_id: str              # unique run id (used for report filename)
    history: List[Dict[str, str]]
    workspace: str              # persistent RAG namespace
    provider: str               # together | groq
    model: str                  # model name for provider
    ocr_backend: str            # paddle | tesseract | auto
    do_web_search: bool         # enable Tavily fusion

    # Current input
    user_input: str
    files: List[Dict[str, Any]] # [{name, bytes}]
    retrieval_events: List[Dict[str, Any]]

    # Router decision and intermediate artifacts
    route: str                  # text_only | uploads_only | mixed
    parsed_docs: List[Dict[str, Any]]  # [{id, name, text, meta}]
    retrieved_chunks: List[str]
    web_snippets: List[str]
    citations: List[str]

    # Schematic artifacts (NEW)
    schematic_json: Dict[str, Any]   # structured {components,nets,texts}
    schematic_summary: str           # short human summary from vision
    part_ctx: str                    # concatenated datasheet snippets for detected parts
    # Schematic control & ephemeral docs (NEW)
    treat_images_as_schematics: bool   # UI checkbox
    has_schematic: bool                # set True if this turn ingested schematic images
    vision_docs: List[str]             # fresh vision texts (summary + JSON) for this turn
    
    # Demo/trace artifacts  👈 ADDED
    trace: List[Dict[str, Any]]
    # ---- LLM Inspector (NEW) ----
    llm_events: List[Dict[str, Any]]  # per-call telemetry
    
    # User (NEW)
    user: str 
    multi_queries: List[str]
    retrieval_pool: List[List[Tuple[str,Dict[str,Any],float,str]]]
    fused_hits: List[Tuple[str,Dict[str,Any],float,str]]
    reranked_hits: List[Tuple[str,Dict[str,Any],float,str]]
    mmr_hits: List[Tuple[str,Dict[str,Any],float,str]]

    # Output and control flags
    final_answer: str
    need_clarification: bool
    end_session: bool
    report_path: str

# Termination tokens (explicit end required by your spec)
TERMINATE = {"ok", "exit", "finish"}

### Helpers ###
def _tok_estimate(text: str) -> int:
    """Fast, dependency-free token estimate (~4 chars/token)."""
    if not text:
        return 0
    n = len(text)
    return max(1, (n + 3) // 4)

def log_llm_event(state: State, *, node: str, model: str,
                  prompt_str: str, output_str: str,
                  usage: Optional[Dict[str, Any]], ms: float) -> None:
    """Append a single LLM call record into state['llm_events'] for the inspector."""
    ptok = int(usage.get("prompt_tokens", 0)) if usage else None
    ctok = int(usage.get("completion_tokens", 0)) if usage else None
    if ptok is None: ptok = _tok_estimate(prompt_str)
    if ctok is None: ctok = _tok_estimate(output_str)

    def _preview(s: str, k=300):
        s = (s or "").strip().replace("\n", " ")
        return (s[:k] + "…") if len(s) > k else s

    (state.setdefault('llm_events', [])).append({
        "node": node,
        "model": model,
        "latency_ms": int(ms),
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "total_tokens": ptok + ctok,
        "prompt_preview": _preview(prompt_str),
        "output_preview": _preview(output_str),
    })



# =====================
# Graph Nodes (pure functions over State)
# =====================

def node_session(state: State) -> State:
    """Check for explicit termination and initialize defaults if missing."""
    t0 = time.perf_counter()
    txt = (state.get('user_input') or '').strip().lower()
    state['end_session'] = txt in TERMINATE
    # clear previous turn’s trace
    state['trace'] = []

    # Ensure defaults present (useful if UI omits)
    state['workspace'] = state.get('workspace') or DEFAULT_WS
    state['provider']  = (state.get('provider') or PROVIDER).lower()
    if state['provider'] != "groq":
        state['provider'] = "groq"
    state['model'] = state.get('model') or MODEL_G
    state['ocr_backend'] = state.get('ocr_backend') or OCR_BACKEND
    state['do_web_search'] = bool(state.get('do_web_search', tavily is not None))
    state['schematic_json'] = state.get('schematic_json') or {}
    state['schematic_summary'] = state.get('schematic_summary') or ""
    state['part_ctx'] = state.get('part_ctx') or ""
    state['treat_images_as_schematics'] = bool(state.get('treat_images_as_schematics'))
   
    # --- Reset per-run ephemeral fields so old values don't echo into this turn ---
    # These are populated by nodes during *this* run only.
    state["parsed_docs"] = []
    state["vision_docs"] = []
    state["retrieval_events"] = []
    state["llm_events"] = []
    state["web_snippets"] = []
    state["citations"] = []
    state["retrieved_chunks"] = []
    state["has_schematic"] = False
    # Additional hygiene to avoid carry-over UI artifacts and stale schematic context
    state.pop("report_path", None) 
    state["schematic_json"] = {}         # clear any prior parsed schematic
    state["schematic_summary"] = ""      # clear prior summary
    state["part_ctx"] = ""               # clear prior datasheet snippets tied to a schematic
    state.pop("route", None)                # cosmetic: router will set this fresh each run
    
    trace_push(state, "session", t0, f"end_session={state['end_session']}")
    return state


def node_router(state: State) -> State:
    """Classify input into text_only / uploads_only / mixed."""
    t0 = time.perf_counter()
    files = state.get('files') or []
    has_pdf = any(f['name'].lower().endswith('.pdf') for f in files)
    has_docx = any(f['name'].lower().endswith('.docx') for f in files)
    has_img = any(f['name'].lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')) for f in files)
    kinds = sum([has_pdf, has_docx, has_img])
    if kinds == 0:
        state['route'] = 'text_only'
    elif kinds == 1 and not (state.get('user_input') or '').strip():
        state['route'] = 'uploads_only'
    else:
        state['route'] = 'mixed'
    trace_push(state, "router", t0, f"route={state['route']}")
    return state

def node_ingest(state: State) -> State:
    """Parse and index uploaded files into the persistent workspace collection (Hybrid).
       PDF → page-first, then sub-chunk long pages.
       DOCX → heading-aware sections, then sub-chunk long sections.
       Images in schematic mode → Vision (no OCR). Other images → OCR and chunk if needed.
    """
    t0 = time.perf_counter()
    display_only: List[Dict[str, Any]] = []               # for "Indexed files" UI (no DB writes)
    chunked_docs: List[Tuple[str, str, Dict[str, Any]]] = []  # to persist in Chroma
    extra_docs: List[Tuple[str, str, Dict[str, Any]]] = []    # vision JSON/summary
    ocr_backend = state.get('ocr_backend') or OCR_BACKEND
    vision_texts: List[str] = []  # fresh vision docs for this turn
    schematic_mode = bool(state.get('treat_images_as_schematics'))

    progress = _get_progress_hook()
    files_list = state.get('files') or []
    total = max(1, len(files_list))

    for i, f in enumerate(files_list):
        name, data = f['name'], f['bytes']
        if progress:
            frac = i / total
            progress(frac, desc=f"Indexing: {name}")
        low = name.lower()
        is_img = low.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

        # --- PDF: page-first then sub-chunk ---
        if low.endswith('.pdf'):
            pages = parse_pdf_pages(data, ocr_backend=ocr_backend)
            total_chars = sum(len(p or "") for p in pages)
            display_only.append({'id': f"{name}-{i}", 'name': name, 'chars': total_chars})

            for pidx, ptxt in enumerate(pages, start=1):
                txt = (ptxt or "").strip()
                if not txt:
                    continue
                if len(txt) <= 2000:
                    body = f"[filename:{name}]\n[page:{pidx}]\n{txt}"
                    meta = {"filename": name, "doc_type": "pdf", "page": pidx}
                    chunked_docs.append((f"{name}-{i}-p{pidx}", body, meta))
                else:
                    for j, ch in enumerate(chunk_text(txt, size=1200, overlap=180), start=1):
                        body = f"[filename:{name}]\n[page:{pidx}] [chunk:{j}]\n{ch}"
                        meta = {"filename": name, "doc_type": "pdf", "page": pidx, "chunk": j}
                        chunked_docs.append((f"{name}-{i}-p{pidx}-c{j}", body, meta))

        # --- DOCX: heading-aware sections then sub-chunk ---
        elif low.endswith('.docx'):
            sections = parse_docx_sections(data)
            total_chars = sum(len(t or "") for _sec, t in sections)
            display_only.append({'id': f"{name}-{i}", 'name': name, 'chars': total_chars})

            for sidx, (spath, stext) in enumerate(sections, start=1):
                txt = (stext or "").strip()
                if not txt:
                    continue
                if len(txt) <= 1500:
                    body = f"[filename:{name}]\n[section:{spath}]\n{txt}"
                    meta = {"filename": name, "doc_type": "docx", "section_path": spath}
                    chunked_docs.append((f"{name}-{i}-s{sidx}", body, meta))
                else:
                    for j, ch in enumerate(chunk_text(txt, size=1200, overlap=180), start=1):
                        body = f"[filename:{name}]\n[section:{spath}] [chunk:{j}]\n{ch}"
                        meta = {"filename": name, "doc_type": "docx", "section_path": spath, "chunk": j}
                        chunked_docs.append((f"{name}-{i}-s{sidx}-c{j}", body, meta))

        # --- Images: schematic mode → Vision; else OCR+chunk if needed ---
        elif is_img and schematic_mode:
            # Vision path (no OCR record stored)
            t_vision = time.perf_counter()
            j, summary, raw, vusage = call_vision_extract(data, model=VISION_MODEL_T)
            dt = (time.perf_counter() - t_vision) * 1000.0
            _prompt = f"[vision] Extract JSON from schematic image: {name}"
            _out = (summary or raw or "")[:1000]
            log_llm_event(state, node="ingest", model=VISION_MODEL_T,
                          prompt_str=_prompt, output_str=_out, usage=vusage, ms=dt)

            display_only.append({'id': f"{name}-{i}", 'name': name, 'chars': 0})
            # (A) Flattened, embedding-friendly summary
            if summary:
                parts = []
                try:
                    for c in (j or {}).get("components", []):
                        pn = c.get("part_number") or c.get("label") or ""
                        if pn:
                            parts.append(str(pn))
                except Exception:
                    pass
                flat = "[doc:vision_summary]\n[fname:%s]\ncomponents: %s\nsummary: %s" % (
                    name, (", ".join(parts) if parts else "(unknown)"), summary
                )
                extra_docs.append((
                    f"{name}-{i}-vision-summary",
                    flat,
                    {"filename": name, "doc_type": "vision_summary", "parts": ", ".join(parts) if parts else ""}
                ))
                vision_texts.append(flat)
            # (B) Raw JSON as text
            if j:
                try:
                    jtxt = json.dumps(j, separators=(",", ":"))
                    parts = [(c.get("part_number") or c.get("label") or "") for c in j.get("components", [])]
                except Exception:
                    jtxt, parts = "{}", []
                jdoc = "[doc:vision_json]\n[fname:%s]\n%s" % (name, jtxt)
                extra_docs.append((
                    f"{name}-{i}-vision-json",
                    jdoc,
                    {"filename": name, "doc_type": "vision_json", "parts": ", ".join(parts) if parts else "",
                     "nets_count": len((j or {}).get("nets", []))}
                ))
                vision_texts.append(jdoc)

        else:
            # generic image or other: OCR as before; chunk if long
            text = parse_image(data, ocr_backend=ocr_backend) if is_img else ""
            total_chars = len(text or "")
            display_only.append({'id': f"{name}-{i}", 'name': name, 'chars': total_chars})
            txt = (text or "").strip()
            if not txt:
                continue
            if len(txt) <= 2000:
                body = f"[filename:{name}]\n{text}"
                meta = {"filename": name, "doc_type": "image_ocr" if is_img else "text"}
                chunked_docs.append((f"{name}-{i}", body, meta))
            else:
                for j, ch in enumerate(chunk_text(txt, size=1200, overlap=180), start=1):
                    body = f"[filename:{name}] [chunk:{j}]\n{ch}"
                    meta = {"filename": name, "doc_type": "image_ocr" if is_img else "text", "chunk": j}
                    chunked_docs.append((f"{name}-{i}-c{j}", body, meta))

    # Persist (DB writes)
    col = get_collection(state['workspace'], state.get('user'))
    if chunked_docs:
        add_docs(col, chunked_docs)
    if extra_docs:
        add_docs(col, extra_docs)

    # ✅ Final tick so the bar reaches 100%
    if progress:
        progress(1.0, desc="Indexing finished")

    state['parsed_docs'] = display_only      # ⬅️ UI only (for "Indexed files" summary)
    state['vision_docs'] = vision_texts
    state['has_schematic'] = state.get('has_schematic') or bool(vision_texts)
    trace_push(state, "ingest", t0, f"indexed={len(display_only)} files, chunks={len(chunked_docs)} extra={len(extra_docs)}")
    return state

def node_multiquery(state: State) -> State:
    t0 = time.perf_counter()
    q = (state.get('user_input') or '').strip()
    provider = state['provider']; model = state['model']
    prompt = f"""Rewrite the question into {MQ_N} diverse, *meaning-preserving* variants for better retrieval.
- Keep critical tokens (part numbers, pins, units) intact when present.
- Vary synonyms, scope (component vs system), and likely datasheet phrasing.
Return one variant per line, no numbering.

Q: {q}"""
    msgs = [{"role":"system","content":"You generate retrieval variants only."},
            {"role":"user","content":prompt}]
    text, usage = chat_llm(msgs, provider=provider, model=model, max_tokens=1500)
    variants = [v.strip(" •\t").strip() for v in (text or "").splitlines() if v.strip()]
    # always include the original at the front (dedup later)
    variants = [q] + [v for v in variants if v.lower() != q.lower()]
    variants = variants[:MQ_N+1]
    state['multi_queries'] = variants
    log_llm_event(state, node="multiquery", model=model,
                  prompt_str="\n".join([m["content"] for m in msgs]),
                  output_str="\n".join(variants), usage=usage, ms=(time.perf_counter()-t0)*1000)
    trace_push(state, "multiquery", t0, f"count={len(variants)}")
    return state

def node_retrieve_pool(state: State) -> State:
    t0 = time.perf_counter()
    col = get_collection(state['workspace'], state.get('user'))
    qs = state.get('multi_queries') or [(state.get('user_input') or '')]
    pooled: List[List[Tuple[str,Dict[str,Any],float,str]]] = []
    for qi in qs:
        docs, metas, dists, ids = retrieve_info(col, qi, k=POOL_K_EACH)
        pooled.append(list(zip(docs, metas, dists, ids)))
    state['retrieval_pool'] = pooled
    trace_push(state, "retrieve_pool", t0, f"lists={len(pooled)} x {POOL_K_EACH}")
    return state

def node_rrf_fuse(state: State) -> State:
    t0 = time.perf_counter()
    pooled = state.get('retrieval_pool') or []
    fused = rrf_fuse(pooled, k=FUSE_K, K=60)
    state['fused_hits'] = fused
    trace_push(state, "rrf_fuse", t0, f"kept={len(fused)}")
    return state

def node_rerank(state: State) -> State:
    t0 = time.perf_counter()
    q = state.get('user_input') or ''
    fused = state.get('fused_hits') or []
    reranked = rerank_cross_encoder(q, fused, model_name=RERANK_MODEL)
    state['reranked_hits'] = reranked
    trace_push(state, "rerank", t0, f"top={min(10,len(reranked))}")
    return state

def node_mmr(state: State) -> State:
    t0 = time.perf_counter()
    q = state.get('user_input') or ''
    reranked = state.get('reranked_hits') or []
    mmr = mmr_select(q, reranked, keep=MMR_KEEP, diversity_lambda=MMR_LAMBDA)
    state['mmr_hits'] = mmr
    trace_push(state, "mmr", t0, f"kept={len(mmr)}")
    return state

def node_parent_promote(state: State) -> State:
    t0 = time.perf_counter()
    col = get_collection(state['workspace'], state.get('user'))
    mmr = state.get('mmr_hits') or []

    # Parent promotion (add limited siblings)
    promoted = promote_parents(col, mmr, max_siblings_per_parent=PROMOTE_SIBLINGS)

    # If this turn produced vision docs, put them first and ensure they’re in hits
    vdocs = state.get('vision_docs') or []
    hits_texts = [d for d, _m, _s, _id in promoted]
    if vdocs:
        hits_texts = list(vdocs) + hits_texts

    # Build retrieval_events (table)
    events: List[Dict[str,Any]] = []
    rank = 1
    if vdocs:
        for vd in vdocs:
            events.append({"rank": 0, "score": None, "filename": "(vision)", "doc_type": "vision",
                           "page": None, "section": None, "preview": vd.replace("\n"," ")[:220]})
    for (doc, meta, score, _id) in promoted:
        m = meta or {}
        filename = os.path.basename(str(m.get("filename") or m.get("source") or m.get("name") or "")) or "(unknown)"
        kind = m.get("doc_type") or ("pdf" if filename.lower().endswith(".pdf") else ("docx" if filename.lower().endswith(".docx") else "text"))
        page = m.get("page"); section = m.get("section_path")
        preview = (doc or "").replace("\n"," ")[:220]
        events.append({"rank": rank, "score": (None if score is None else round(float(score),3)),
                       "filename": filename, "doc_type": kind, "page": page, "section": section, "preview": preview})
        rank += 1

    state['retrieval_events'] = events
    state['retrieved_chunks'] = hits_texts
    trace_push(state, "parent_promote", t0, f"hits={len(hits_texts)}")
    return state

# def node_retrieve(state: State) -> State:
#     """Retrieve top-k chunks from the workspace collection given the current question."""
#     t0 = time.perf_counter()
#     q = state.get('user_input') or ''
#     col = get_collection(state['workspace'], state.get('user'))

#     docs, metas, dists, ids = retrieve_info(col, q, k=8)  # a few more since chunks are smaller with Hybrid
#     events: List[Dict[str, Any]] = []
#     for rank, (doc, meta, dist, _id) in enumerate(zip(docs, metas, dists, ids), 1):
#         meta = meta or {}
#         filename = os.path.basename(str(meta.get("filename") or meta.get("source") or meta.get("name") or "")) or "(unknown)"
#         kind = meta.get("doc_type") or ("pdf" if filename.lower().endswith(".pdf") else ("docx" if filename.lower().endswith(".docx") else "text"))
#         page = meta.get("page")
#         section = meta.get("section_path")
#         preview = (doc or "").replace("\n", " ")[:220]
#         # Chroma gives a distance; convert to a similarity-ish score (optional)
#         score = None
#         try:
#             score = 1.0 - float(dist) if dist is not None else None
#         except Exception:
#             score = None
#         events.append({
#             "rank": rank,
#             "score": (None if score is None else round(score, 3)),
#             "filename": filename,
#             "doc_type": kind,
#             "page": page,
#             "section": section,
#             "preview": preview,
#         })

#     # If this turn produced vision docs, show them first and ensure they are in hits
#     vdocs = state.get('vision_docs') or []
#     if vdocs:
#         v_events = []
#         for vd in vdocs:
#             v_events.append({
#                 "rank": 0,
#                 "score": None,
#                 "filename": "(vision)",
#                 "doc_type": "vision",
#                 "page": None,
#                 "section": None,
#                 "preview": vd.replace("\n", " ")[:220],
#             })
#         state['retrieval_events'] = v_events + events
#         hits = list(vdocs) + list(docs or [])
#     else:
#         state['retrieval_events'] = events
#         hits = docs or []

#     state['retrieved_chunks'] = hits
#     trace_push(state, "retrieve", t0, f"hits={len(hits or [])}")
#     return state


def node_web(state: State) -> State:
    """Optional web search via Tavily; stores snippets and raw citations."""
    t0 = time.perf_counter()
    state['web_snippets'] = []
    state['citations'] = []
    if not (state.get('do_web_search') and tavily):
        trace_push(state, "web", t0, "skipped")
        return state
    q = state.get('user_input') or ''
    if not q.strip():
        trace_push(state, "web", t0, "skipped")
        return state
    try:
        res = tavily.search(q, max_results=5)
        snippets: List[str] = []
        cites: List[str] = []
        for item in res.get('results', [])[:5]:
            title = item.get('title') or 'source'
            url = item.get('url') or ''
            content = (item.get('content') or '')[:900]
            snippets.append(f"[{title}] {content}")
            if url:
                cites.append(url)
        state['web_snippets'] = snippets
        state['citations'] = cites
        trace_push(state, "web", t0, f"sources={len(cites)}")
    except Exception as e:
        state['web_snippets'] = [f"[web search error] {e}"]
        trace_push(state, "web", t0, "error")
    return state

def parts_expand(state: State) -> State:
    """From the retrieved vision JSON/summary, extract part numbers and fetch datasheet context."""
    t0 = time.perf_counter()
    chunks = state.get('retrieved_chunks') or []
    json_text = ""
    summary_text = ""
    # pick the first matching blocks
    for c in chunks:
        if "[doc:vision_json]" in c:
            json_text = c
            break
    for c in chunks:
        if "[doc:vision_summary]" in c:
            summary_text = c
            break

    # parse JSON (strip tags + find {...})
    j = {}
    try:
        body = json_text.split("\n", 2)[-1] if json_text else ""
        start = body.find("{"); end = body.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(body[start:end+1])
    except Exception:
        j = {}

    # extract unique part numbers (fallback to label if pn missing)
    pns: List[str] = []
    for c in j.get("components", []):
        pn = c.get("part_number") or c.get("label") or ""
        pn = str(pn).strip()
        if pn and pn not in pns:
            pns.append(pn)

    # retrieve a few chunks per PN from the local collection
    col = get_collection(state['workspace'], state.get('user'))
    part_ctx_segments: List[str] = []
    for pn in pns:
        hits = retrieve(col, pn, k=3)
        if hits:
            part_ctx_segments.append(f"[{pn}] " + "\n".join(hits))

    state['schematic_json'] = j
    state['schematic_summary'] = summary_text
    state['part_ctx'] = "\n\n".join(part_ctx_segments)
    trace_push(state, "parts_expand", t0, f"parts={len(pns)} ctx_segs={len(part_ctx_segments)}")
    return state

def schematic_answer(state: State) -> State:
    """Use the structured JSON + part_ctx to produce a precise circuit explanation."""
    t0 = time.perf_counter()
    provider = state['provider']
    model    = state['model']

    j = state.get('schematic_json') or {}
    part_ctx = state.get('part_ctx') or ""
    summary  = state.get('schematic_summary') or ""
    user_q   = state.get('user_input') or ""

    # keep it compact but explicit
    sys = (
        f"You are a senior hardware design assistant. Default language: {DEFAULT_LANGUAGE}. "
        "You are given structured {components, nets} extracted from a schematic, plus datasheet snippets. "
        "Respond with:\n"
        "1) Chips list (ref, part, role)\n"
        "2) Key connections (Net: ref.pin → ref.pin)\n"
        "3) Functional description (concise)\n"
        "4) Any assumptions/uncertainties\n"
        "Quote short phrases from provided context when you rely on them."
    )

    j_txt = json.dumps(j, indent=2) if j else "{}"
    ctx = []
    if summary: ctx.append(summary)
    if part_ctx: ctx.append(part_ctx)
    rag_ctx = "\n\n".join(ctx)

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Question: {user_q}\n\nStructured JSON:\n{j_txt}\n\nContext:\n{rag_ctx}\n\nBe specific; use refdes and pin names. If something is unknown, say so."},
    ]

    # --- Citations policy + list (reuse web citations if any) ---
    cits = state.get("citations") or []
    allowed_urls = [u for u in cits if isinstance(u, str) and u.startswith("http")]
    cit_lines = [f"[{i+1}] {u}" for i, u in enumerate(allowed_urls)]
    citations_block = "\n".join(cit_lines)

    policy = (
        "CITATIONS POLICY:\n"
        "- Do NOT output a standalone 'Sources' or 'References' section.\n"
        "- If you cite, use bracketed indices like [1], [2] referring ONLY to the CITATIONS list provided.\n"
        "- If no CITATIONS are provided, do not invent links.\n"
        f"CITATIONS:\n{citations_block if cit_lines else '(none)'}"
    )
    # Prepend policy to steer the model
    messages.insert(0, {"role": "system", "content": policy})

    reply,usage = chat_llm(messages, provider=provider, model=model, max_tokens=2000, temperature=0.2)
    # scrub bogus 'Sources' and any stray URLs not in our allowed list
    reply = sanitize_answer(reply, allowed_urls) # ✅ removes any “Sources:” footer / stray URLs
    state['final_answer'] = reply
    dt = (time.perf_counter() - t0) * 1000.0
    # compact prompt string for inspector (text parts only)
    _prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    log_llm_event(state, node="schematic_answer", model=model,
                  prompt_str=_prompt_str, output_str=reply, usage=usage, ms=dt)

    pt = usage.get("prompt_tokens","~"); ct = usage.get("completion_tokens","~")
    trace_push(state, "schematic_answer", t0, f"model={model}  in={pt}  out={ct}")
    return state


def node_answer(state: State) -> State:
    """Compose system+user messages with RAG/Web context and call the model."""
    t0 = time.perf_counter()
    provider = state['provider']
    model    = state['model']

    rag_ctx = "\n\n".join(state.get('retrieved_chunks') or [])
    web_ctx = "\n\n".join(state.get('web_snippets') or [])

    sys = (
        f"You are a senior hardware design assistant. Default language: {DEFAULT_LANGUAGE}. "
        "Be precise; show formulas/units; cite short snippets (in quotes) when you rely on context. "
        "If uncertain, ask for page/figure/pin/net names you need."
    )

    user_q = state.get('user_input') or ''
    if not user_q.strip():
        state['final_answer'] = (
            "I indexed your files into the workspace. Ask a question (e.g., 'Compute bias current for Q1', 'Check the op-amp stability with R_L=…')."
        )
        trace_push(state, "answer", t0, f"model={model} (no question)")
        return state

    ctx = f"""RAG context (uploaded/private):
{rag_ctx or "(none)"}

Web context (Tavily):
{web_ctx or "(none)"}"""

    messages = [
        {"role": "system", "content": sys},
        {"role": "user",   "content": f"""Question:
{user_q}

Use context if helpful; do not hallucinate.

{ctx}"""},
    ]

    # --- Citations policy + list (only if web citations exist) ---
    cits = state.get("citations") or []
    allowed_urls = [u for u in cits if isinstance(u, str) and u.startswith("http")]
    cit_lines = [f"[{i+1}] {u}" for i, u in enumerate(allowed_urls)]
    citations_block = "\n".join(cit_lines)

    policy = (
        "CITATIONS POLICY:\n"
        "- Do NOT output a standalone 'Sources' or 'References' section.\n"
        "- If you cite, use bracketed indices like [1], [2] referring ONLY to the CITATIONS list provided.\n"
        "- If no CITATIONS are provided, do not invent links.\n"
        f"CITATIONS:\n{citations_block if cit_lines else '(none)'}"
    )

    # Prepend the policy as a system message to discipline output
    messages.insert(0, {"role": "system", "content": policy})


    reply, usage = chat_llm(messages, provider=provider, model=model)
    reply = sanitize_answer(reply, allowed_urls)

    state['final_answer'] = reply
    dt = (time.perf_counter() - t0) * 1000.0
    _prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    log_llm_event(state, node="answer", model=model,
                  prompt_str=_prompt_str, output_str=reply, usage=usage, ms=dt)

    pt = usage.get("prompt_tokens","~"); ct = usage.get("completion_tokens","~")
    trace_push(state, "answer", t0, f"model={model}  in={pt}  out={ct}")
    
    return state

def node_verify(state: State) -> State:
    """Light heuristic verification → if weak context/answer, request clarifications."""
    t0 = time.perf_counter()
    ans = state.get('final_answer') or ''
    chunks = state.get('retrieved_chunks') or []
    web = state.get('web_snippets') or []
    state['need_clarification'] = (len(chunks) < 1 and len(web) < 1) or (len(ans) < 120)
    trace_push(state, "verify", t0, "clarify" if state['need_clarification'] else "ok")
    return state


def inventory_docs(state: State) -> State:
    """List previously uploaded PDFs/DOCX in the current project."""
    t0 = time.perf_counter()
    col = get_collection(state['workspace'], state.get('user'))
    res = col.get(include=["metadatas"])
    seen = set()
    for m in (res.get("metadatas") or []):
        m = m or {}
        if (m.get("doc_type") in ("pdf", "docx")):
            fn = m.get("filename") or m.get("source") or m.get("name") or ""
            fn = os.path.basename(str(fn))
            if fn:
                seen.add(fn)
    files = sorted(seen, key=str.lower)
    n = len(files)
    bullets = "\n".join(f"- {f}" for f in files) or "(none found)"
    state["final_answer"] = f"You have **{n}** document(s) (PDF/DOCX) in this project:\n\n{bullets}"
    trace_push(state, "inventory_docs", t0, f"count={n}")
    return state

def inventory_schematics(state: State) -> State:
    """List previously uploaded schematics recognized via Vision (by filename)."""
    t0 = time.perf_counter()
    col = get_collection(state['workspace'], state.get('user'))
    res = col.get(include=["metadatas"])
    seen = set()
    for m in (res.get("metadatas") or []):
        m = m or {}
        # Schematic artifacts are stored as 'vision_summary' and 'vision_json'
        if m.get("doc_type") in ("vision_summary", "vision_json"):
            fn = m.get("filename") or m.get("source") or m.get("name") or ""
            fn = os.path.basename(str(fn))
            if fn:
                seen.add(fn)
    files = sorted(seen, key=str.lower)
    n = len(files)
    bullets = "\n".join(f"- {f}" for f in files) or "(none found)"
    state["final_answer"] = f"You have **{n}** schematic image(s) in this project:\n\n{bullets}"
    trace_push(state, "inventory_schematics", t0, f"count={n}")
    return state

def node_clarify(state: State) -> State:
    """Ask targeted follow-ups to resolve ambiguity (pins, refs, conditions)."""
    t0 = time.perf_counter()
    provider = state['provider']
    model    = state['model']
    user_q   = state.get('user_input') or ''

    messages = [
        {"role": "system", "content": "Ask up to 3 precise clarification questions to answer a hardware design query."},
        {"role": "user", "content": f"""We need more details to answer this question accurately:
{user_q}

Focus your questions on: component part numbers, voltage/current conditions, pin/net names, figure/page numbers, test points."""},
    ]

    reply, usage = chat_llm(messages, provider=provider, model=model, max_tokens=5000)
    state['final_answer'] = reply

    dt = (time.perf_counter() - t0) * 1000.0
    _prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    log_llm_event(state, node="clarify", model=model,
                  prompt_str=_prompt_str, output_str=reply, usage=usage, ms=dt)

    trace_push(state, "clarify", t0, "follow-ups asked")
    return state


def node_report(state: State) -> State:
    """Generate a Markdown session summary and save it to a temp file for download."""
    t0 = time.perf_counter()
    provider = state['provider']
    model    = state['model']
    hist = state.get('history', [])
    last_turns = "\n".join([f"{m['role']}: {m['content']}" for m in hist[-12:]])

    messages = [
        {"role": "system", "content": "Summarize the session into Markdown with: bullet points, equations if present, cited snippets, and TODOs."},
        {"role": "user",   "content": f"""Summarize this hardware design assistant session.

{last_turns}"""}
    ]
    summary_md, usage = chat_llm(messages, provider=provider, model=model, max_tokens=5000)

    path = os.path.join(tempfile.gettempdir(), f"hwqa_summary_{state.get('thread_id','x')}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    state['final_answer'] = "Session ended. A Markdown report has been generated (see download below).\n\n" + summary_md
    state['report_path'] = path

    dt = (time.perf_counter() - t0) * 1000.0
    _prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    log_llm_event(state, node="report", model=model,
                  prompt_str=_prompt_str, output_str=summary_md, usage=usage, ms=dt)

    trace_push(state, "report", t0, "summary generated")
    return state


# =====================
# Graph wiring (this is the showcase part)
# =====================

def cond_end(state: State) -> str:
    return 'end' if state.get('end_session') else 'continue'


def cond_route(state: State) -> str:
    return state.get('route','text_only')

# --- Simple intent detectors for inventory commands ---
def _want_inventory_docs(q: str) -> bool:
    return (q or "").strip().lower() == "list docs"

def _want_inventory_schematics(q: str) -> bool:
    return (q or "").strip().lower() == "list schematics"

def cond_after_retrieve(state: State) -> str:
    """Decide next step after retrieval, returning one of:
       'inv_docs' | 'inv_schematics' | 'schematic' | 'with_web' | 'direct'
    """
    hits = state.get("retrieved_chunks") or []
    q = state.get("user_input") or ""

    # 🔎 Inventory intents (fast-exit)
    if _want_inventory_schematics(q):
        return "inv_schematics"
    if _want_inventory_docs(q):
        return "inv_docs"

    # Schematic path wins if we ingested/recognized a schematic this turn
    try:
        if state.get("has_schematic"):
            return "schematic"
    except Exception:
        pass

     # ✅ New rule: when the toggle is ON, always visit the web node next.
    if bool(state.get("do_web_search")):
        return "with_web"

    # Otherwise keep the existing local-only heuristic
    if len(hits) >= 2:
        return "direct"

    # Otherwise, still go direct (the answer node may ask for clarification later)
    return "direct"


def cond_verify(state: State) -> str:
    return 'clarify' if state.get('need_clarification') else 'ok'


def build_graph():
    g = StateGraph(State)

    # Register nodes
    g.add_node('session',   node_session)
    g.add_node('router',    node_router)
    g.add_node('ingest',    node_ingest)
    g.add_node('multiquery',      node_multiquery)
    g.add_node('retrieve_pool',   node_retrieve_pool)
    g.add_node('rrf_fuse',        node_rrf_fuse)
    g.add_node('rerank',          node_rerank)
    g.add_node('mmr',             node_mmr)
    g.add_node('parent_promote',  node_parent_promote)

    g.add_node('web',       node_web)
    g.add_node('answer',    node_answer)
    g.add_node('verify',    node_verify)
    g.add_node('clarify',   node_clarify)
    g.add_node('report',    node_report)
    g.add_node('parts_expand', parts_expand)
    g.add_node('schematic_answer', schematic_answer)
    g.add_node('inventory_docs', inventory_docs)
    g.add_node('inventory_schematics', inventory_schematics)



    # Entry point → either end (report) or continue to router
    g.set_entry_point('session')
    g.add_conditional_edges('session', cond_end, {
        'end': 'report',
        'continue': 'router',
    })

    # Router → either straight to retrieve (text_only) or ingest first (uploads)
    g.add_conditional_edges('router', cond_route, {
        'text_only': 'multiquery',
        'uploads_only': 'ingest',
        'mixed': 'ingest',
    })

    # After ingest, go multiquery; then maybe web; then answer; then verify
    g.add_edge('ingest', 'multiquery')
    # Retrieval chain
    g.add_edge('multiquery', 'retrieve_pool')
    g.add_edge('retrieve_pool', 'rrf_fuse')
    g.add_edge('rrf_fuse', 'rerank')
    g.add_edge('rerank', 'mmr')
    g.add_edge('mmr', 'parent_promote')

    g.add_conditional_edges('parent_promote', cond_after_retrieve, {
        'inv_docs': 'inventory_docs',
        'inv_schematics': 'inventory_schematics',
        'schematic': 'parts_expand',
        'with_web': 'web',
        'direct': 'answer',
    })

    g.add_edge('web', 'answer')
    g.add_edge('parts_expand', 'schematic_answer')
    g.add_edge('schematic_answer', 'verify')
    g.add_edge('inventory_docs', 'verify')
    g.add_edge('inventory_schematics', 'verify')
    g.add_edge('answer', 'verify')

    # Verify can ask for clarifications or finish the turn
    g.add_conditional_edges('verify', cond_verify, {
        'clarify': 'clarify',
        'ok': END,
    })
    g.add_edge('clarify', END)
    g.set_finish_point('clarify')

    # MemorySaver shows you know checkpointing; for persistence across turns in a server, swap for a DB-backed store
    mem = MemorySaver()
    return g.compile(checkpointer=mem)

# Build the compiled app once (import‑time)
APP = build_graph()

# =====================
# Gradio UI — simple but expressive for demos
# =====================

def to_pairs(history: List[Dict[str, str]]):
    """Convert role-wise history into Chatbot (user, assistant) pairs."""
    pairs: List[Tuple[str, str]] = []
    u: Optional[str] = None
    for m in history:
        if m['role'] == 'user':
            u = m['content']
        elif m['role'] == 'assistant' and u is not None:
            pairs.append((u, m['content']))
            u = None
    return pairs

def run_once(request: gr.Request,
             thread_id: str,
             chat_pairs: List[Tuple[str, str]],
             msg: str,
             files: List[gr.File],
             project: str,
             provider: str,
             model: str,
             ocr_backend: str,
             do_web_search: bool,
             treat_images_as_schematics: bool,
             progress=None):
    """Execute one graph turn; user is derived from the JWT cookie; workspace = selected project."""
    # Who is logged in?
    user_email = get_logged_in_email_from_request(request) or "anonymous"

    # Rehydrate conversation history into role messages
    history: List[Dict[str, str]] = []
    for u_msg, a_msg in (chat_pairs or []):
        history += [
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": a_msg},
        ]

    # Read uploaded files as bytes now (Gradio gives temp filepaths)
    uploads: List[Dict[str, Any]] = []
    for f in files or []:
        try:
            with open(f.name, 'rb') as fp:
                uploads.append({"name": os.path.basename(f.name), "bytes": fp.read()})
        except Exception:
            pass

    # Initial state for this turn
    init: State = {
        'thread_id': thread_id,
        'history': history,
        'user_input': msg or '',
        'files': uploads,
        'workspace': project or DEFAULT_WS,
        'user': user_email,
        'provider': (provider or PROVIDER).lower(),
        'model': model or MODEL_G,
        'ocr_backend': ocr_backend or OCR_BACKEND,
        'do_web_search': bool(do_web_search),
        'treat_images_as_schematics': bool(treat_images_as_schematics),
        'trace': []  # reset per turn so MemorySaver doesn’t carry old steps
    }

    # Run the graph (one pass)
    _set_progress_hook(progress)          # make the hook visible to nodes, but not in State
    out: State = APP.invoke(init, config={"configurable": {"thread_id": thread_id}})

    # Prepare reply content
    reply = out.get('final_answer') or '(no reply)'
    history.append({"role": "user", "content": msg})

    # If we ingested files this turn, prepend a tiny indexing summary for transparency
    parsed = out.get('parsed_docs') or []
    prefix = ""
    if parsed:
        prefix = "Indexed files:\n" + "\n".join([f"• {d['name']} ({d.get('chars', len(d.get('text','')))} chars)" for d in parsed]) + "\n\n"


    history.append({"role": "assistant", "content": prefix + reply})

    # Build artifacts for the Trace tab
    trace_md, _trace_json, dag_img = render_trace_artifacts(out)

    # Expose a Markdown report file if the session ended this turn
    report_path = out.get('report_path', None)
    return to_pairs(history), (report_path if report_path else None), trace_md, dag_img

def build_gradio_blocks(auth_secret: Optional[str] = None):
    # 👇 Force the same secret that FastAPI used to sign the JWT
    global AUTH_SECRET
    if auth_secret:
        AUTH_SECRET = auth_secret
    print("[debug] Gradio AUTH_SECRET prefix:", AUTH_SECRET[:8])
    
    with gr.Blocks(title="Hardware QA Assistant — Advanced LangGraph", css=UPLOADER_CSS, theme="freddyaboulton/dracula_revamped") as demo:
        gr.Markdown("# Hardware QA Assistant — Advanced LangGraph 🔧Developer: Antonios Karvelas")
        gr.Markdown("Upload datasheets/schematics; ask questions. Type **OK / exit / finish** to end and get a Markdown report.")

        thread = gr.State(value=os.urandom(4).hex())

        with gr.Tabs():
            with gr.Tab("Chat"):
                # --- User bar ---
                user_md = gr.Markdown("User: _(not logged in)_", elem_id="user_label")
                logout_html = gr.HTML('<div style="text-align:right;"><a href="/logout">Logout</a></div>')

                # --- Workspace Manager (per-user) ---
                with gr.Accordion("Workspace Manager (per-user projects)", open=False):
                    with gr.Row():
                        project_dd = gr.Dropdown(choices=[], label="Project", allow_custom_value=True)
                        refresh_btn = gr.Button("Refresh list")
                        new_name = gr.Textbox(label="New project name")
                        new_btn = gr.Button("Create")
                        del_btn = gr.Button("Delete selected")

                # --- Model / OCR / Web ---
                with gr.Row():
                    provider  = gr.Dropdown(choices=["groq"], value=PROVIDER, label="LLM Provider")
                    model     = gr.Textbox(value=MODEL_G, label="Model name")  # Groq text model
                def _on_provider_change(prov: str):
                    return gr.update(value=MODEL_G)
                provider.change(_on_provider_change, inputs=provider, outputs=model)
                with gr.Row():
                    ocr_backend = gr.Dropdown(choices=["paddle","tesseract","auto"], value=OCR_BACKEND, label="OCR backend")
                    do_web_search = gr.Checkbox(value=True if tavily else False, label="Use Tavily web search if needed")

                chat = gr.Chatbot(height=560, type="tuples", elem_id="chatui")
                # Uploader + Vision checkbox side-by-side (compact)

                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about a circuit/component… e.g., 'Is the LDO stable with 10µF and ESR=50mΩ?'  |  Tip: type 'List docs' (PDF/DOCX) or 'List schematics' (images).",
                    info="Tip: 'List docs' → list uploaded documents (PDF/DOCX).  'List schematics' → list uploaded schematic images."
                )
                send = gr.Button("Send")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=160):
                        files = gr.File(
                            label="Upload PDFs / DOCX / images",
                            file_count="multiple",
                            container=False,          # trims extra vertical padding
                            elem_id="uploader"        # CSS hook
                        )
                    with gr.Column(scale=1, min_width=120):
                        treat_as_schematic = gr.Checkbox(
                            value=False,
                            label="Treat images as schematics (use Vision)",
                            container=False,                # <-- trims padding
                            elem_id="schem_tick"            # <-- CSS hook below
                        )

                

                summary = gr.File(label="Session summary (Markdown)", visible=False)

            with gr.Tab("Trace"):
                trace_md = gr.Markdown(value="(Run a query to see the last execution trace here.)", elem_id="trace_md")
                dag_img = gr.Image(label="Graph (executed path highlighted)", type="pil")
                
            with gr.Tab("Manage PDF files"):
                gr.Markdown("### Delete PDFs from the current Project\n"
                            "Pick the files you want to remove from retrieval (Chroma).")
                with gr.Row():
                    gr.Markdown("Current **Project** selection above will be used.")
                    refresh_files_btn = gr.Button("Refresh file list")
                files_ck = gr.CheckboxGroup(label="Files in project", choices=[], interactive=True)
                with gr.Row():
                    delete_files_btn = gr.Button("Delete selected", variant="stop")
                    manage_status = gr.Markdown()

                # Wire handlers
                def _ui_refresh_files(request: gr.Request, ws):
                    email = get_logged_in_email_from_request(request)
                    choices = list_workspace_files_by_filename(ws, email)
                    return gr.update(choices=choices, value=[])

                def _ui_delete_files(request: gr.Request, ws, picked):
                    email = get_logged_in_email_from_request(request)
                    msg = delete_files_by_filename(ws, email, picked or [])
                    return msg

                refresh_files_btn.click(_ui_refresh_files, inputs=project_dd, outputs=files_ck)
                delete_files_btn.click(_ui_delete_files, inputs=[project_dd, files_ck], outputs=manage_status) \
                                .then(_ui_refresh_files, inputs=project_dd, outputs=files_ck)


            with gr.Tab("Report"):
                gr.Markdown("### View generated report(s)\n"
                            "Load the latest or choose a specific `hwqa_summary_*.md` and render it below.")
                with gr.Row():
                    rep_refresh = gr.Button("Refresh list")
                    rep_load_latest = gr.Button("Load latest")
                    rep_picker = gr.Dropdown(label="Select a report file", choices=[], interactive=True)
                report_view = gr.Markdown(value="_No report loaded yet._")

                def _rep_refresh():
                    files = list_summary_reports()
                    return gr.update(choices=files, value=(files[0] if files else None))

                def _rep_load_latest():
                    files = list_summary_reports()
                    if not files:
                        return gr.update(value="_No reports found._")
                    return read_report_text(files[0])

                def _rep_load_pick(path):
                    return read_report_text(path)

                rep_refresh.click(_rep_refresh, inputs=None, outputs=rep_picker)
                rep_load_latest.click(_rep_load_latest, inputs=None, outputs=report_view)
                rep_picker.change(_rep_load_pick, inputs=rep_picker, outputs=report_view)



        # --- Helpers bound to UI ---
        def ui_refresh(request: gr.Request):
            email = get_logged_in_email_from_request(request) or "(not logged in)"
            projects = list_projects_for_user(email) if "@" in email else []
            # pick default if available
            value = projects[0] if projects else ""
            return gr.update(choices=projects, value=value), f"**User:** {email}"

        def ui_create(request: gr.Request, name: str):
            email = get_logged_in_email_from_request(request)
            if not (email and name.strip()):
                # keep the user label unchanged but tell them why
                label = f"**User:** {email or '(not logged in)'}  \nCreate failed: missing email or project name."
                return gr.update(), label
            # touching the collection creates it if missing
            _ = get_collection(name, email)
            projects = list_projects_for_user(email)
            # update dropdown (choices + select the new project) AND refresh the user label
            label = f"**User:** {email}  \nCreated project **{_slug(name)}**."
            return gr.update(choices=projects, value=name), label

        def ui_delete(request: gr.Request, name: str):
            email = get_logged_in_email_from_request(request)
            if not (email and name):
                label = f"**User:** {email or '(not logged in)'}  \nDelete failed."
                return gr.update(), label

            ok = delete_project(email, name)
            msg = f"Deleted project **{name}**." if ok else "Delete failed or project not found."
            projects = list_projects_for_user(email)
            label = f"**User:** {email}  \n{msg}"
            return gr.update(choices=projects, value=(projects[0] if projects else "")), label



        # add `progress=gr.Progress(track_tqdm=True)` in the parameters
        def on_send(request: gr.Request, t, c, m, f, proj, prov, mdl, ocr, web, schem,
            progress=gr.Progress(track_tqdm=True)):
            chat_pairs, summary_path, trace_md_val, dag_img_val = \
                run_once(request, t, c, m, f, proj, prov, mdl, ocr, web, schem, progress=progress)

            upload_reset = gr.update(value=None)
            msg_reset = gr.update(value="")

            # 👇 show the file widget only when a report was generated this turn
            summary_upd = (gr.update(value=summary_path, visible=True)
                        if summary_path else gr.update(value=None, visible=False))

            return chat_pairs, summary_upd, trace_md_val, dag_img_val, upload_reset, msg_reset


        refresh_btn.click(ui_refresh, inputs=None, outputs=[project_dd, user_md])
        new_btn.click(ui_create, inputs=[new_name], outputs=[project_dd, user_md], preprocess=False)
        del_btn.click(ui_delete, inputs=[project_dd], outputs=[project_dd, user_md], preprocess=False)

        send.click(on_send,
            inputs=[thread, chat, msg, files, project_dd, provider, model, ocr_backend, do_web_search, treat_as_schematic],
            outputs=[chat, summary, trace_md, dag_img, files, msg],
            concurrency_limit="default", concurrency_id="chat"
        )
        msg.submit(on_send,
            inputs=[thread, chat, msg, files, project_dd, provider, model, ocr_backend, do_web_search, treat_as_schematic],
            outputs=[chat, summary, trace_md, dag_img, files, msg],
            concurrency_limit="default", concurrency_id="chat"
        )

        # Logout (call FastAPI /logout)
        # def do_logout():
        #     import requests
        #     try:
        #         requests.post("/logout", timeout=3)
        #     except Exception:
        #         pass
        #     return gr.update(value="**User:** (logged out)")

        # logout_btn.click(do_logout, inputs=None, outputs=user_md)
         # 🔧 Auto-load the user & project list on page load (must be INSIDE Blocks)
        demo.load(ui_refresh, inputs=None, outputs=[project_dd, user_md])
    demo.queue(default_concurrency_limit=1, max_size=32)  # serialize requests; prevents parallel actions during ingest
    return demo

def launch():
    # keep standalone mode for local runs (non-FastAPI)
    demo = build_gradio_blocks()
    demo.queue().launch(share=True)


if __name__ == '__main__':
    launch()
