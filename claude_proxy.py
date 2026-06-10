"""
Claude Code / Codex aware proxy.

Default behaviour: transparent forwarding to UPSTREAM (currently litellm).

Special cases:

1. Claude Code emits a sub-request whenever it wants to run a WebSearch tool.
   The sub-request has a recognisable shape — intercepted here and handled
   via search API, returning synthetic Anthropic SSE.

2. Codex includes a `type: web_search` tool in `/v1/responses` requests.
   When the model calls this tool, we intercept the response, execute the
   search via Ollama or Tavily (env-controlled), then send a follow-up
   request so the model can incorporate the results.
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

UPSTREAM = os.environ.get("UPSTREAM", "http://litellm:4000")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")
LOG_DIR = Path(os.environ.get("LOG_DIR", "/var/log/claude-proxy"))
LOG_REQUESTS = os.environ.get("LOG_REQUESTS", "1").strip().lower() not in ("0", "false", "no", "off")
if LOG_REQUESTS:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_SEARCH_URL = os.environ.get(
    "OLLAMA_SEARCH_URL", "https://ollama.com/api/web_search"
)
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))
SEARCH_SNIPPET_LEN = int(os.environ.get("SEARCH_SNIPPET_LEN", "1500"))

# Codex web_search backend: "ollama" (default) or "tavily"
CODEX_SEARCH_BACKEND = os.environ.get("CODEX_SEARCH_BACKEND", "ollama").strip().lower()
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = os.environ.get("TAVILY_SEARCH_URL", "https://api.tavily.com/search")

WEB_SEARCH_SYSTEM_MARKER = "You are an assistant for performing a web search tool use"
WEB_SEARCH_QUERY_PREFIX = "Perform a web search for the query: "


app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))


# ---------------------- logging helpers ----------------------

def _log_req(rid: str, headers: dict, body: bytes) -> None:
    if not LOG_REQUESTS:
        return
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
    if not LOG_REQUESTS:
        return
    (LOG_DIR / f"{rid}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )


# ---------------------- Claude Code web_search detection ----------------------

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


# ---------------------- search backends ----------------------

async def run_ollama_search(query: str) -> list:
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


async def run_tavily_search(query: str) -> list:
    headers = {
        "Content-Type": "application/json",
    }
    body = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": SEARCH_MAX_RESULTS,
        "include_answer": False,
    }
    try:
        r = await client.post(TAVILY_SEARCH_URL, headers=headers, json=body, timeout=30.0)
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


async def run_search(query: str) -> list:
    """Execute search via the configured backend (ollama or tavily)."""
    if CODEX_SEARCH_BACKEND == "tavily" and TAVILY_API_KEY:
        return await run_tavily_search(query)
    return await run_ollama_search(query)


def _format_search_results_text(hits: list) -> str:
    """Format search hits into a text block for the model to consume."""
    if not hits:
        return "No search results found."
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] {h.get('title', '')}")
        if h.get("url"):
            parts.append(f"    URL: {h['url']}")
        if h.get("content"):
            parts.append(f"    {h['content']}")
    return "\n".join(parts)


# ---------------------- Claude Code SSE synthesis ----------------------

def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def synthesize_search_sse(model: str, query: str, rid: str):
    hits = await run_ollama_search(query)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"
    in_tokens = max(1, len(query) // 4)
    out_tokens = max(1, sum(len((h.get("title") or "") + (h.get("url") or "")) for h in hits) // 4)

    if not LOG_REQUESTS: return
    (LOG_DIR / f"{rid}.synthetic.json").write_text(
        json.dumps({"query": query, "tool_use_id": tool_use_id, "hits": hits},
                   ensure_ascii=False, indent=2)
    )

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "model": model or "web-search", "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": 0},
        },
    })

    yield _sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {
            "type": "server_tool_use", "id": tool_use_id,
            "name": "web_search", "input": {},
        },
    })
    yield _sse("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": json.dumps({"query": query}, ensure_ascii=False)},
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

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
        "type": "content_block_start", "index": 1,
        "content_block": {
            "type": "web_search_tool_result", "tool_use_id": tool_use_id,
            "content": result_content,
        },
    })
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 1})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})
    yield b"data: [DONE]\n\n"


# ---------------------- Codex web_search tool loop ----------------------

def _is_responses_path(full_path: str) -> bool:
    path = "/" + full_path.strip("/")
    return path in {"/v1/responses", "/responses"}


def _find_web_search_call(output: list) -> Optional[dict]:
    """Find a function_call for web_search in the response output."""
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and item.get("name") == "web_search":
            return item
    return None


def _without_web_search_tools(tools: Any) -> list:
    """Return upstream-ready tools that can remain available after web_search completes."""
    if not isinstance(tools, list):
        return []
    kept = []
    for tool in tools:
        if not isinstance(tool, dict):
            kept.append(tool)
            continue
        if tool.get("name") == "web_search" or tool.get("type") == "web_search":
            continue
        kept.append(tool)
    return kept


def _sanitize_responses_input_for_upstream(input_items: Any, request_annotations: dict) -> Any:
    """Convert or remove Codex Responses output-only items that Ollama/LiteLLM cannot accept as input.

    Codex clients may send these item types back in subsequent turns as part of
    conversation history. Upstream backends only understand a subset of item
    types (message, function_call, function_call_output, reasoning). We must:

    - custom_tool_call       -> convert to function_call (preserves name/call_id/arguments)
    - custom_tool_call_output -> convert to function_call_output (preserves call_id/output)
    - tool_search_call       -> strip (metadata only, no useful content for the model)
    - tool_search_output     -> strip (metadata only, no useful content for the model)
    - web_search_call         -> strip (search metadata only; actual results are in
                                 function_call_output from our proxy's follow-up)
    """
    if not isinstance(input_items, list):
        return input_items
    sanitized = []
    stripped = {}
    for item in input_items:
        if not isinstance(item, dict):
            sanitized.append(item)
            continue
        item_type = item.get("type")
        if item_type == "web_search_call":
            stripped[item_type] = stripped.get(item_type, 0) + 1
            continue
        if item_type == "custom_tool_call":
            # Convert to function_call: the model needs to know it previously called this tool.
            # custom_tool_call.input may not be valid JSON (e.g. apply_patch patches are plain
            # text starting with "*** Begin Patch"); wrap as a JSON object in that case so the
            # upstream backend can parse the arguments field.
            raw_input = item.get("input", "{}")
            if isinstance(raw_input, str):
                try:
                    json.loads(raw_input)
                    arguments = raw_input
                except (json.JSONDecodeError, ValueError):
                    arguments = json.dumps({"_raw_input": raw_input}, ensure_ascii=False)
            else:
                arguments = json.dumps(raw_input, ensure_ascii=False)
            converted = {
                "type": "function_call",
                "call_id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": arguments,
            }
            sanitized.append(converted)
            request_annotations.setdefault("codex_input_converted", {})
            request_annotations["codex_input_converted"]["custom_tool_call"] = request_annotations["codex_input_converted"].get("custom_tool_call", 0) + 1
            continue
        if item_type == "custom_tool_call_output":
            # Convert to function_call_output: preserve the tool result
            converted = {
                "type": "function_call_output",
                "call_id": item.get("call_id", ""),
                "output": item.get("output", ""),
            }
            sanitized.append(converted)
            request_annotations.setdefault("codex_input_converted", {})
            request_annotations["codex_input_converted"]["custom_tool_call_output"] = request_annotations["codex_input_converted"].get("custom_tool_call_output", 0) + 1
            continue
        if item_type in ("tool_search_call", "tool_search_output"):
            stripped[item_type] = stripped.get(item_type, 0) + 1
            continue
        sanitized.append(item)
    if stripped:
        request_annotations["codex_input_items_stripped"] = stripped
    return sanitized


async def _execute_codex_search_and_followup(
    first_resp: dict, payload: dict, rid: str, started: float, request_annotations: dict,
    user_auth: str = "",
) -> dict:
    """Handle the Codex web_search tool loop:
    1. Extract query from function_call
    2. Execute search
    3. Send follow-up request with results
    4. Merge into a complete response with web_search_call output item
    """
    ws_call = _find_web_search_call(first_resp.get("output", []))
    if not ws_call:
        return first_resp

    # Parse query from function_call arguments
    try:
        args = json.loads(ws_call.get("arguments", "{}"))
    except Exception:
        args = {}
    query = args.get("query", "")
    if not query:
        request_annotations["codex_web_search_error"] = "empty query"
        return first_resp

    # Execute search
    search_started = time.time()
    hits = await run_search(query)
    search_text = _format_search_results_text(hits)

    request_annotations["codex_web_search"] = {
        "query": query,
        "backend": CODEX_SEARCH_BACKEND,
        "hits": len(hits),
        "search_ms": int((time.time() - search_started) * 1000),
    }

    # Log search results
    if not LOG_REQUESTS: return
    (LOG_DIR / f"{rid}.codex_search.json").write_text(
        json.dumps({"query": query, "backend": CODEX_SEARCH_BACKEND, "hits": hits},
                   ensure_ascii=False, indent=2)
    )

    # Construct follow-up request: include original input + function_call_output.
    # Codex may send output-only items such as web_search_call back in later turns;
    # strip those before forwarding to Ollama/LiteLLM.
    call_id = ws_call.get("call_id", "functions.web_search:0")
    followup_input = list(_sanitize_responses_input_for_upstream(payload.get("input", []), request_annotations) or [])

    # Add the model's function_call as a completed item
    followup_input.append({
        "type": "function_call",
        "call_id": call_id,
        "name": "web_search",
        "arguments": ws_call.get("arguments", "{}"),
    })
    # Add the function call output with search results
    followup_input.append({
        "type": "function_call_output",
        "call_id": call_id,
        "output": search_text,
    })

    # Build follow-up request. Drop web_search itself because the search has
    # already completed, but preserve other upstream-ready tools (notably the
    # flattened MCP function tools) so the model can continue a same-turn tool
    # chain after reading search results.
    followup_payload = {
        "model": payload.get("model"),
        "input": followup_input,
        "stream": False,
    }
    followup_tools = _without_web_search_tools(payload.get("tools"))
    if followup_tools:
        followup_payload["tools"] = followup_tools
        request_annotations["codex_followup_tools"] = len(followup_tools)
    if payload.get("instructions"):
        followup_payload["instructions"] = payload["instructions"]
    if payload.get("temperature") is not None:
        followup_payload["temperature"] = payload["temperature"]

    followup_body = json.dumps(followup_payload, ensure_ascii=False).encode("utf-8")
    followup_url = f"{UPSTREAM.rstrip('/')}/v1/responses"

    followup_headers = {"Content-Type": "application/json"}
    auth_key = user_auth or UPSTREAM_API_KEY
    if auth_key:
        followup_headers["Authorization"] = f"Bearer {auth_key}"

    followup_started = time.time()
    try:
        r = await client.post(followup_url, headers=followup_headers, content=followup_body, timeout=300.0)
        r.raise_for_status()
        second_resp = r.json()
    except Exception as e:
        request_annotations["codex_followup_error"] = repr(e)
        fallback = dict(first_resp)
        fallback["status"] = "completed"
        fallback["output"] = [
            item for item in first_resp.get("output", [])
            if isinstance(item, dict) and item.get("type") == "reasoning"
        ]
        fallback["output"].append({
            "id": f"ws_{uuid.uuid4().hex[:24]}",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": query},
        })
        fallback["output"].append({
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": search_text,
                "annotations": [],
            }],
        })
        return fallback

    request_annotations["codex_followup_ms"] = int((time.time() - followup_started) * 1000)

    # Merge: take the second response as the base, but prepend web_search_call
    ws_output_item = {
        "id": f"ws_{uuid.uuid4().hex[:24]}",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "query": query,
        },
    }

    # Combine outputs: keep original reasoning + web_search_call + second response outputs
    merged_output = []

    # Keep reasoning from first response
    for item in first_resp.get("output", []):
        if item.get("type") == "reasoning":
            merged_output.append(item)

    # Add web_search_call
    merged_output.append(ws_output_item)

    # Add all output from second response (the model's final answer)
    for item in second_resp.get("output", []):
        merged_output.append(item)

    # Build merged response using second_resp as base
    merged = dict(second_resp)
    merged["output"] = merged_output
    # Accumulate usage
    u1 = first_resp.get("usage") or {}
    u2 = second_resp.get("usage") or {}
    # Merge usage: sum numeric fields, keep nested dicts from second response
    def _merge_usage(a, b):
        result = {}
        for k in set(list(a.keys()) + list(b.keys())):
            va, vb = a.get(k), b.get(k)
            if isinstance(va, dict) or isinstance(vb, dict):
                result[k] = vb if vb is not None else va
            else:
                result[k] = (va or 0) + (vb or 0)
        return result

    merged["usage"] = _merge_usage(u1, u2)

    return merged


def _responses_sse_event(data_line: str) -> Optional[dict]:
    """Parse a single SSE data line into a dict. Returns None if not parseable."""
    line = data_line.strip()
    if line.startswith("data: "):
        payload = line[6:]
        if payload == "[DONE]":
            return {"type": "done"}
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


async def _buffer_responses_stream(upstream_resp) -> bytes:
    """Buffer the entire upstream SSE response into bytes."""
    chunks = []
    async for chunk in upstream_resp.aiter_raw():
        chunks.append(chunk)
    await upstream_resp.aclose()
    return b"".join(chunks)


def _parse_sse_output(raw: bytes) -> list:
    """Extract output items from a buffered SSE stream."""
    output_items = []
    text_data = raw.decode("utf-8", errors="replace")
    for line in text_data.split("\n"):
        ev = _responses_sse_event(line)
        if not ev:
            continue
        if ev.get("type") == "response.output_item.done":
            item = ev.get("item")
            if item:
                output_items.append(item)
        elif ev.get("type") == "response.completed":
            resp = ev.get("response", {})
            if resp.get("output"):
                output_items = resp["output"]
    return output_items


def _convert_function_call_to_namespaced(
    output_items: list, mcp_namespace_map: dict
) -> list:
    """Rewrite function_call items that match flattened MCP tools so the Codex
    client can route them to the correct MCP server.

    When we flatten a namespace like mcp__firecrawl → mcp__firecrawl__firecrawl_search,
    the Ollama model generates a function_call with name="mcp__firecrawl__firecrawl_search"
    and namespace=None.  The Codex client's tool router (build_tool_call) handles
    ResponseItem::FunctionCall by constructing ToolName::new(namespace, name).  If the
    namespace is present, it creates a namespaced ToolName that matches the registered
    MCP handler.  If absent, it creates a plain ToolName that won't match.

    IMPORTANT: We must keep type="function_call" (NOT "custom_tool_call") because
    the CustomToolCall branch uses ToolName::plain(name) which IGNORES the namespace
    field entirely, causing "unsupported custom tool call" errors.

    Codex 0.135 expects the namespace exactly as advertised in the request
    tools array, e.g. "mcp__chrome_devtools". Adding the MCP delimiter here
    makes the router look up a different tool name and produces unsupported call
    errors.
    """
    if not mcp_namespace_map:
        return output_items
    converted = []
    for item in output_items:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        if item.get("type") != "function_call":
            converted.append(item)
            continue
        name = item.get("name", "")
        mapping = mcp_namespace_map.get(name)
        if not mapping:
            converted.append(item)
            continue
        # Keep as function_call but set name and namespace correctly.
        # Codex router expects the namespace to match the request namespace,
        # e.g. "mcp__server" rather than "mcp__server__".
        new_item = dict(item)
        new_item["name"] = mapping["name"]           # e.g. "click"
        new_item["namespace"] = mapping["namespace"]  # e.g. "mcp__chrome_devtools"
        # Keep: type="function_call", call_id, id, arguments, status
        converted.append(new_item)
    return converted


def _convert_mcp_calls_in_response(
    resp: dict, mcp_namespace_map: dict, request_annotations: dict
) -> dict:
    """Walk a full Responses API dict and rewrite function_call items that match
    flattened MCP tools, setting the correct namespace so the Codex client
    can route them to the right MCP server."""
    if not mcp_namespace_map:
        return resp
    output = resp.get("output", [])
    new_output = _convert_function_call_to_namespaced(output, mcp_namespace_map)
    changed = len(new_output) != len(output) or any(
        a is not b for a, b in zip(new_output, output)
    )
    if changed:
        resp = dict(resp)
        resp["output"] = new_output
        request_annotations.setdefault("mcp_call_converted", 0)
        request_annotations["mcp_call_converted"] += 1
    return resp


def _build_sse_from_response(resp: dict) -> bytes:
    """Convert a complete response dict into SSE stream bytes."""
    parts = []

    # response.created
    parts.append(f"data: {json.dumps({'type': 'response.created', 'response': resp}, ensure_ascii=False)}\n\n")

    # output items
    for i, item in enumerate(resp.get("output", [])):
        parts.append(f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': i, 'item': item}, ensure_ascii=False)}\n\n")
        parts.append(f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': i, 'item': item}, ensure_ascii=False)}\n\n")

    # in_progress
    parts.append(f"data: {json.dumps({'type': 'response.in_progress', 'response': resp}, ensure_ascii=False)}\n\n")

    # completed
    parts.append(f"data: {json.dumps({'type': 'response.completed', 'response': resp}, ensure_ascii=False)}\n\n")

    parts.append("data: [DONE]\n\n")

    return "".join(parts).encode("utf-8")


# ---------------------- proxy core ----------------------

@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(full_path: str, request: Request):
    rid = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    started = time.time()
    request_annotations = {}

    body = await request.body()
    req_headers = {k: v for k, v in request.headers.items()}
    _log_req(rid, req_headers, body)

    payload: Optional[dict] = None
    if body:
        try:
            payload = json.loads(body)
        except Exception:
            payload = None

    # --- Intercept Claude Code WebSearch sub-requests ---
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
                "intercepted": "claude_code_web_search",
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

    # --- Codex request sanitization ---
    is_codex_responses = (
        request.method == "POST"
        and isinstance(payload, dict)
        and _is_responses_path(full_path)
    )
    has_web_search = False

    # Track flattened namespace → sub-tool mapping so we can rewrite
    # function_call responses with the correct namespace for the Codex client.
    # mcp_namespace_map: {"mcp__ns__tool_name": {"namespace": "mcp__ns", "name": "tool_name"}}
    mcp_namespace_map: dict = {}

    if request.method == "POST" and isinstance(payload, dict):
        # Process tools: flatten namespace into function tools, convert web_search
        tools = payload.get("tools")
        if isinstance(tools, list):
            new_tools = []
            for t in tools:
                if not isinstance(t, dict):
                    new_tools.append(t)
                    continue
                ttype = t.get("type")
                if ttype == "namespace":
                    # Flatten namespace tools into top-level function tools.
                    # Ollama strips the inner "tools" array from namespace objects,
                    # so the model can't see sub-tools inside a namespace.
                    # By flattening, each sub-tool becomes a standalone function the
                    # model can see and call.  We record the mapping so that when
                    # the model calls "mcp__ns__tool", we can rewrite the response
                    # function_call with the correct namespace (which the Codex client
                    # needs to route the call to the right MCP server).
                    ns_name = t.get("name", "")
                    ns_desc = t.get("description", "")
                    sub_tools = t.get("tools") or []
                    if isinstance(sub_tools, list) and sub_tools:
                        for sub in sub_tools:
                            if not isinstance(sub, dict) or sub.get("type") != "function":
                                continue
                            flat = dict(sub)
                            inner_name = sub.get("name", "")
                            flat_name = f"{ns_name}__{inner_name}"
                            flat["name"] = flat_name
                            if ns_desc:
                                flat["description"] = (
                                    f"[{ns_name}] {ns_desc}\n\n{sub.get('description', '')}"
                                )
                            new_tools.append(flat)
                            # Record mapping for response conversion
                            mcp_namespace_map[flat_name] = {
                                "namespace": ns_name,
                                "name": inner_name,
                            }
                        request_annotations.setdefault("codex_tools_flattened", 0)
                        request_annotations["codex_tools_flattened"] += 1
                    else:
                        # No sub-tools — keep the namespace as-is so the model
                        # at least sees the name and description.
                        new_tools.append(t)
                elif ttype == "web_search":
                    # Convert to a standard function tool so the model knows how to call it
                    has_web_search = True
                    new_tools.append({
                        "type": "function",
                        "name": "web_search",
                        "description": "Search the web for current information. Use this tool whenever you need up-to-date information, news, or facts beyond your training data.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string"
                                }
                            },
                            "required": ["query"]
                        }
                    })
                    request_annotations["codex_web_search_converted"] = True
                else:
                    new_tools.append(t)
            payload["tools"] = new_tools

        # Strip Responses output-only items that are not valid input for Ollama/LiteLLM.
        if is_codex_responses and isinstance(payload.get("input"), list):
            payload["input"] = _sanitize_responses_input_for_upstream(payload.get("input"), request_annotations)

        # Downgrade unsupported reasoning effort values (e.g. "xhigh" -> "high")
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if isinstance(effort, str) and effort not in {"high", "medium", "low", "max", "none"}:
                reasoning["effort"] = "high"
                request_annotations["reasoning_downgraded"] = f"{effort}->high"
        elif isinstance(reasoning, str) and reasoning not in {"high", "medium", "low", "max", "none"}:
            payload["reasoning"] = "high"
            request_annotations["reasoning_downgraded"] = f"{reasoning}->high"

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        # DEBUG: Save the actual body being sent upstream
        if LOG_REQUESTS:
            (LOG_DIR / f"{rid}.upstream.body").write_bytes(body)
            tool_names = [t.get("name") for t in payload.get("tools", []) if isinstance(t, dict)]
            (LOG_DIR / f"{rid}.upstream.tools.txt").write_text(
            json.dumps(tool_names, ensure_ascii=False, indent=2)
        )

    # --- Forward to upstream ---
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
    # If the request body was modified (e.g. Codex namespace tools flattened,
    # input items sanitized), we need to re-serialize the modified payload and
    # send THAT to upstream, not the original body bytes.
    modified = request_annotations.get("codex_tools_flattened") or request_annotations.get("codex_input_items_stripped") or request_annotations.get("codex_input_converted")
    if modified and payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
            **request_annotations,
            "elapsed_ms": int((time.time() - started) * 1000),
        })
        return Response(content=f"upstream error: {e}", status_code=502)

    resp_headers = dict(upstream_resp.headers)
    if LOG_REQUESTS:
        (LOG_DIR / f"{rid}.resp.headers.json").write_text(
        json.dumps(resp_headers, ensure_ascii=False, indent=2)
    )

    ctype = upstream_resp.headers.get("content-type", "")
    is_stream = "text/event-stream" in ctype or "stream" in ctype

    # --- Codex web_search tool loop ---
    # Buffering strategy:
    #   - If request has NO web_search tool → stream through directly (no buffering)
    #   - If request HAS web_search tool → MUST buffer because we need to check if the
    #     model called web_search and, if so, execute the search and send a follow-up.
    #
    # In theory we could stream-while-scanning SSE events and detect function_call:web_search
    # mid-stream, then stop and execute the search. However, this requires either:
    #   (a) Not sending any events to the client until we know if web_search was called
    #       (which is effectively buffering), or
    #   (b) Sending early events and then "switching" to a merged response, which breaks
    #       the SSE protocol since we can't undo already-sent events.
    #
    # The practical optimization already applied: requests WITHOUT web_search tool bypass
    # buffering entirely and stream through with low latency. Codex always includes web_search
    # in its tool list, but if the model doesn't call it, we still buffer. This is unavoidable
    # because we must see the complete response to confirm no web_search call was made.
    needs_web_search_intercept = is_codex_responses and upstream_resp.status_code == 200 and has_web_search

    # For Codex /v1/responses requests with flattened MCP tools, we must buffer
    # the response to rewrite function_call items with the correct namespace
    # so the Codex client can route MCP tool calls to the right MCP server.
    # The Codex router (build_tool_call) uses ResponseItem::FunctionCall →
    # ToolName::new(namespace, name) → ToolName::namespaced(ns, name) to look up
    # the MCP handler. We must keep type="function_call" (NOT "custom_tool_call")
    # because CustomToolCall uses ToolName::plain(name) which ignores namespace.
    needs_mcp_call_conversion = (
        is_codex_responses
        and upstream_resp.status_code == 200
        and mcp_namespace_map
        and not needs_web_search_intercept  # web_search path handles this separately
    )

    if needs_mcp_call_conversion:
        # Buffer the entire upstream response to rewrite function_call namespace
        raw_body = await _buffer_responses_stream(upstream_resp)
        if LOG_REQUESTS:
            (LOG_DIR / f"{rid}.resp.body").write_bytes(raw_body)

        if is_stream:
            output_items = _parse_sse_output(raw_body)
            # Reconstruct a full response from SSE events to do the conversion
            # We need to find the response.completed event to get the base response dict
            resp_dict = {}
            text_data = raw_body.decode("utf-8", errors="replace")
            for line in text_data.split("\n"):
                ev = _responses_sse_event(line)
                if ev and ev.get("type") == "response.completed":
                    resp_dict = ev.get("response", {})
                    break
            if resp_dict:
                resp_dict = _convert_mcp_calls_in_response(
                    resp_dict, mcp_namespace_map, request_annotations
                )
                sse_bytes = _build_sse_from_response(resp_dict)
                # DEBUG: Save the converted response for inspection
                if LOG_REQUESTS:
                    (LOG_DIR / f"{rid}.resp.client.sse").write_bytes(sse_bytes)
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": 200,
                    "is_stream": True,
                    "content_type": ctype,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=sse_bytes,
                    status_code=200,
                    headers={"content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache"},
                    media_type="text/event-stream",
                )
            # Fallback: return raw if we couldn't parse
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": True,
                "content_type": ctype,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers={"content-type": ctype or "text/event-stream; charset=utf-8"},
                media_type=ctype or "text/event-stream",
            )
        else:
            # Non-streaming: parse JSON, convert, return
            try:
                resp_dict = json.loads(raw_body)
            except Exception:
                resp_dict = {}
            if resp_dict:
                resp_dict = _convert_mcp_calls_in_response(
                    resp_dict, mcp_namespace_map, request_annotations
                )
                # DEBUG: Save the converted JSON response for inspection
                (LOG_DIR / f"{rid}.resp.client.json").write_bytes(
                    json.dumps(resp_dict, ensure_ascii=False, indent=2).encode("utf-8")
                )
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": upstream_resp.status_code,
                    "is_stream": False,
                    "content_type": ctype,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=json.dumps(resp_dict, ensure_ascii=False).encode("utf-8"),
                    status_code=upstream_resp.status_code,
                    headers={"content-type": "application/json"},
                    media_type="application/json",
                )
            # Fallback
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": False,
                "content_type": ctype,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            out_headers = {}
            for k, v in resp_headers.items():
                kl = k.lower()
                if kl in ("content-length", "transfer-encoding", "connection"):
                    continue
                out_headers[k] = v
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers=out_headers,
                media_type=ctype or None,
            )

    if needs_web_search_intercept:
        # Buffer the entire upstream response
        raw_body = await _buffer_responses_stream(upstream_resp)
        if LOG_REQUESTS:
            (LOG_DIR / f"{rid}.resp.body").write_bytes(raw_body)

        # Parse output items from the response
        first_resp = None
        if is_stream:
            output_items = _parse_sse_output(raw_body)
        else:
            try:
                first_resp = json.loads(raw_body)
                output_items = first_resp.get("output", [])
            except Exception:
                output_items = []

        # Check if model called web_search
        ws_call = _find_web_search_call(output_items)

        if ws_call:
            # Need to rebuild first_resp for SSE case
            if first_resp is None:
                # Reconstruct from SSE events
                first_resp = {"output": output_items, "status": "completed"}
                # Try to get usage from the completed event
                text_data = raw_body.decode("utf-8", errors="replace")
                for line in text_data.split("\n"):
                    ev = _responses_sse_event(line)
                    if ev and ev.get("type") == "response.completed":
                        first_resp.update(ev.get("response", {}))
                        break

            # Execute search + follow-up
            user_auth = req_headers.get("authorization", "") or req_headers.get("Authorization", "")
            merged = await _execute_codex_search_and_followup(
                first_resp, payload, rid, started, request_annotations, user_auth=user_auth
            )

            # Rewrite MCP function_call namespace in merged response so Codex
            # client can route to the right MCP server.
            if mcp_namespace_map:
                merged = _convert_mcp_calls_in_response(
                    merged, mcp_namespace_map, request_annotations
                )

            # Return merged response
            if is_stream:
                # Re-encode as SSE
                sse_bytes = _build_sse_from_response(merged)
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": 200,
                    "is_stream": True,
                    "content_type": ctype,
                    "codex_web_search_handled": True,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=sse_bytes,
                    status_code=200,
                    headers={"content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache"},
                    media_type="text/event-stream",
                )
            else:
                _log_meta(rid, {
                    "rid": rid,
                    "upstream_url": upstream_url,
                    "method": request.method,
                    "status": 200,
                    "is_stream": False,
                    "content_type": ctype,
                    "codex_web_search_handled": True,
                    **request_annotations,
                    "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                    "elapsed_ms": int((time.time() - started) * 1000),
                })
                return Response(
                    content=json.dumps(merged, ensure_ascii=False).encode("utf-8"),
                    status_code=200,
                    headers={"content-type": "application/json"},
                    media_type="application/json",
                )

        # No web_search call found — still need to convert MCP calls if any
        if mcp_namespace_map:
            if is_stream:
                # Parse SSE, convert, re-encode
                output_items = _parse_sse_output(raw_body)
                resp_dict = {}
                text_data = raw_body.decode("utf-8", errors="replace")
                for line in text_data.split("\n"):
                    ev = _responses_sse_event(line)
                    if ev and ev.get("type") == "response.completed":
                        resp_dict = ev.get("response", {})
                        break
                if resp_dict:
                    resp_dict = _convert_mcp_calls_in_response(
                        resp_dict, mcp_namespace_map, request_annotations
                    )
                    sse_bytes = _build_sse_from_response(resp_dict)
                    _log_meta(rid, {
                        "rid": rid,
                        "upstream_url": upstream_url,
                        "method": request.method,
                        "status": upstream_resp.status_code,
                        "is_stream": True,
                        "content_type": ctype,
                        "codex_buffered": True,
                        **request_annotations,
                        "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                        "elapsed_ms": int((time.time() - started) * 1000),
                    })
                    return Response(
                        content=sse_bytes,
                        status_code=upstream_resp.status_code,
                        headers={"content-type": "text/event-stream; charset=utf-8"},
                        media_type="text/event-stream",
                    )
            else:
                # Non-streaming: parse JSON, convert, return
                try:
                    resp_dict = json.loads(raw_body)
                except Exception:
                    resp_dict = {}
                if resp_dict:
                    resp_dict = _convert_mcp_calls_in_response(
                        resp_dict, mcp_namespace_map, request_annotations
                    )
                    _log_meta(rid, {
                        "rid": rid,
                        "upstream_url": upstream_url,
                        "method": request.method,
                        "status": upstream_resp.status_code,
                        "is_stream": False,
                        "content_type": ctype,
                        "codex_buffered": True,
                        **request_annotations,
                        "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                        "elapsed_ms": int((time.time() - started) * 1000),
                    })
                    out_headers = {}
                    for k, v in resp_headers.items():
                        kl = k.lower()
                        if kl in ("content-length", "transfer-encoding", "connection"):
                            continue
                        out_headers[k] = v
                    return Response(
                        content=json.dumps(resp_dict, ensure_ascii=False).encode("utf-8"),
                        status_code=upstream_resp.status_code,
                        headers=out_headers,
                        media_type=ctype or None,
                    )

        # No web_search call found and no MCP conversion needed — return as-is
        if is_stream:
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": True,
                "content_type": ctype,
                "codex_buffered": True,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers={"content-type": ctype or "text/event-stream; charset=utf-8"},
                media_type=ctype or "text/event-stream",
            )
        else:
            _log_meta(rid, {
                "rid": rid,
                "upstream_url": upstream_url,
                "method": request.method,
                "status": upstream_resp.status_code,
                "is_stream": False,
                "content_type": ctype,
                "codex_buffered": True,
                **request_annotations,
                "upstream_started_offset_ms": int((upstream_started - started) * 1000),
                "elapsed_ms": int((time.time() - started) * 1000),
            })
            out_headers = {}
            for k, v in resp_headers.items():
                kl = k.lower()
                if kl in ("content-length", "transfer-encoding", "connection"):
                    continue
                out_headers[k] = v
            return Response(
                content=raw_body,
                status_code=upstream_resp.status_code,
                headers=out_headers,
                media_type=ctype or None,
            )

    # --- Default: transparent stream forward (non-Codex or error responses) ---
    resp_path = LOG_DIR / f"{rid}.resp.body" if LOG_REQUESTS else None

    async def streamer():
        try:
            async for chunk in upstream_resp.aiter_raw():
                if resp_path is not None:
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
                **request_annotations,
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
