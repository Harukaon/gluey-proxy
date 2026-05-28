"""
Claude Code aware proxy.

Default behaviour: transparent forwarding to UPSTREAM (9router).

Special case: Claude Code emits a sub-request whenever it wants to run a
WebSearch tool. The sub-request has a recognisable shape:

  system contains a text block: "You are an assistant for performing a web search tool use"
  tools contains:               {"type": "web_search_20250305", ...}
  messages[0].content[0].text:  "Perform a web search for the query: <query>"

The model behind 9router does not natively execute web_search_20250305.
Instead of routing the sub-request to the model (which would just emit a
tool_use the client cannot satisfy), we:

  1. Extract <query>
  2. Call https://ollama.com/api/web_search
  3. Stream back a synthetic Anthropic SSE response containing a single text
     block with the formatted search results.

Claude Code then folds that text into the parent conversation as the
WebSearch tool_result, and the next turn flows transparently through the
proxy to 9router as normal.
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM", "http://9router:20128")
LOG_DIR = Path(os.environ.get("LOG_DIR", "/var/log/claude-proxy"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_SEARCH_URL = os.environ.get(
    "OLLAMA_SEARCH_URL", "https://ollama.com/api/web_search"
)
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))
SEARCH_SNIPPET_LEN = int(os.environ.get("SEARCH_SNIPPET_LEN", "1500"))

WEB_SEARCH_SYSTEM_MARKER = "You are an assistant for performing a web search tool use"
WEB_SEARCH_QUERY_PREFIX = "Perform a web search for the query: "

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))


# ---------------------- logging helpers ----------------------

def _log_req(rid: str, headers: dict, body: bytes) -> None:
    (LOG_DIR / f"{rid}.req.headers.json").write_text(
        json.dumps(headers, ensure_ascii=False, indent=2)
    )
    (LOG_DIR / f"{rid}.req.body").write_bytes(body)
    try:
        parsed = json.loads(body) if body else None
        if parsed is not None:
            (LOG_DIR / f"{rid}.req.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2)
            )
    except Exception:
        pass


def _log_meta(rid: str, meta: dict) -> None:
    (LOG_DIR / f"{rid}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )


# ---------------------- web_search detection ----------------------

def _system_has_marker(system_field: Any) -> bool:
    if not system_field:
        return False
    if isinstance(system_field, str):
        return WEB_SEARCH_SYSTEM_MARKER in system_field
    if isinstance(system_field, list):
        for block in system_field:
            if isinstance(block, dict):
                txt = block.get("text", "")
                if isinstance(txt, str) and WEB_SEARCH_SYSTEM_MARKER in txt:
                    return True
    return False


def _tools_has_web_search(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    for t in tools:
        if isinstance(t, dict) and isinstance(t.get("type"), str):
            if t["type"].startswith("web_search_"):
                return True
    return False


def _extract_query(messages: Any) -> Optional[str]:
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    text = None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                break
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith(WEB_SEARCH_QUERY_PREFIX):
        return text[len(WEB_SEARCH_QUERY_PREFIX):].strip()
    return None


def is_websearch_subrequest(payload: dict) -> Optional[str]:
    """Return the query string if payload is a Claude Code web_search sub-request, else None."""
    if not isinstance(payload, dict):
        return None
    if not _system_has_marker(payload.get("system")):
        return None
    if not _tools_has_web_search(payload.get("tools")):
        return None
    return _extract_query(payload.get("messages"))


# ---------------------- ollama search + SSE synthesis ----------------------

async def run_ollama_search(query: str) -> list:
    """Call ollama web_search and return list of {title, url, content}. Never raises."""
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {"query": query, "max_results": SEARCH_MAX_RESULTS}
    try:
        r = await client.post(OLLAMA_SEARCH_URL, headers=headers, json=body, timeout=30.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [{"title": f"Search failed: {e}", "url": "", "content": ""}]

    results = data.get("results", []) if isinstance(data, dict) else []
    out = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if len(content) > SEARCH_SNIPPET_LEN:
            content = content[:SEARCH_SNIPPET_LEN].rstrip() + "..."
        out.append({"title": title, "url": url, "content": content})
    return out


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def synthesize_search_sse(model: str, query: str, rid: str):
    """Yield Anthropic SSE stream containing server_tool_use + web_search_tool_result
    blocks, matching what Claude Code's WebSearchTool client parser expects.

    Reference: cc-haha src/tools/WebSearchTool/WebSearchTool.ts:432-521 — the parser
    only counts blocks of type 'web_search_tool_result' as searches, and pulls
    title/url out of each entry in its content[] array.
    """
    hits = await run_ollama_search(query)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"
    in_tokens = max(1, len(query) // 4)
    out_tokens = max(1, sum(len((h.get("title") or "") + (h.get("url") or "")) for h in hits) // 4)

    # Persist what we built for debugging.
    (LOG_DIR / f"{rid}.synthetic.json").write_text(
        json.dumps({"query": query, "tool_use_id": tool_use_id, "hits": hits},
                   ensure_ascii=False, indent=2)
    )

    # message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model or "web-search",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": 0},
        },
    })

    # Block 0: server_tool_use — declares the model called web_search(query=...)
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "server_tool_use",
            "id": tool_use_id,
            "name": "web_search",
            "input": {},
        },
    })
    yield _sse("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": json.dumps({"query": query}, ensure_ascii=False),
        },
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    # Block 1: web_search_tool_result — the actual hits the client renders.
    # Each entry shape mirrors Anthropic's native format. Claude Code only reads
    # title and url; encrypted_content is included so other consumers don't choke.
    result_content = [
        {
            "type": "web_search_result",
            "title": h.get("title") or "",
            "url": h.get("url") or "",
            "encrypted_content": h.get("content") or "",
            "page_age": None,
        }
        for h in hits
        if h.get("url")
    ]

    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": result_content,
        },
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 1})

    # message_delta / message_stop
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})
    yield b"data: [DONE]\n\n"


# ---------------------- proxy core ----------------------

@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(full_path: str, request: Request):
    rid = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    started = time.time()

    body = await request.body()
    req_headers = {k: v for k, v in request.headers.items()}
    _log_req(rid, req_headers, body)

    payload: Optional[dict] = None
    if body:
        try:
            payload = json.loads(body)
        except Exception:
            payload = None

    # Intercept Claude Code WebSearch sub-requests.
    if (
        request.method == "POST"
        and full_path.endswith("v1/messages")
        and isinstance(payload, dict)
    ):
        query = is_websearch_subrequest(payload)
        if query:
            model = payload.get("model") or "web-search"
            _log_meta(rid, {
                "rid": rid,
                "intercepted": "web_search",
                "query": query,
                "model": model,
                "elapsed_ms_to_decision": int((time.time() - started) * 1000),
            })
            return StreamingResponse(
                synthesize_search_sse(model, query, rid),
                status_code=200,
                headers={"cache-control": "no-cache"},
                media_type="text/event-stream",
            )

    # Default: transparent forward to upstream.
    upstream_url = f"{UPSTREAM.rstrip('/')}/{full_path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    fwd_headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "connection", "transfer-encoding"):
            continue
        fwd_headers[k] = v

    upstream_started = time.time()
    try:
        upstream_resp = await client.send(
            client.build_request(
                request.method,
                upstream_url,
                headers=fwd_headers,
                content=body,
            ),
            stream=True,
        )
    except Exception as e:
        _log_meta(rid, {
            "rid": rid,
            "upstream_url": upstream_url,
            "method": request.method,
            "error": repr(e),
            "elapsed_ms": int((time.time() - started) * 1000),
        })
        return Response(content=f"upstream error: {e}", status_code=502)

    resp_headers = dict(upstream_resp.headers)
    (LOG_DIR / f"{rid}.resp.headers.json").write_text(
        json.dumps(resp_headers, ensure_ascii=False, indent=2)
    )

    resp_path = LOG_DIR / f"{rid}.resp.body"

    ctype = upstream_resp.headers.get("content-type", "")
    is_stream = "text/event-stream" in ctype or "stream" in ctype

    async def streamer():
        try:
            async for chunk in upstream_resp.aiter_raw():
                with open(resp_path, "ab") as f:
                    f.write(chunk)
                yield chunk
        finally:
            await upstream_resp.aclose()
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": is_stream,
                "content_type": ctype,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })

    out_headers = {}
    for k, v in resp_headers.items():
        kl = k.lower()
        if kl in ("content-length", "transfer-encoding", "connection"):
            continue
        out_headers[k] = v

    return StreamingResponse(
        streamer(),
        status_code=upstream_resp.status_code,
        headers=out_headers,
        media_type=ctype or None,
    )


@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()
