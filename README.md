# Gluey Proxy

Gluey Proxy is the compatibility layer we run in front of LiteLLM for coding
clients. It forwards normal traffic to an upstream OpenAI/Anthropic-compatible
gateway and handles tool protocol gaps that upstream providers do not natively
support.

The current production source of truth is the AWS deployment that runs this app
as `claude-proxy`.

## What It Does

- Forwards ordinary requests to `UPSTREAM`.
- Intercepts Claude Code WebSearch subrequests on `/v1/messages` and returns
  synthetic Anthropic SSE using the configured search backend.
- Converts Codex `/v1/responses` `web_search` tools into normal function tools,
  executes search if the model calls it, and sends a follow-up request so the
  model can use the search results.
- Flattens Codex MCP namespace tools before sending to upstream models and
  rewrites returned tool calls so Codex can route them back to the right MCP
  server.
- Logs request/response details under `LOG_DIR` for debugging.

## Environment

Required:

```text
UPSTREAM=http://litellm:4000
UPSTREAM_API_KEY=<liteLLM key used by the proxy>
OLLAMA_API_KEY=<optional, required for Ollama web search>
```

Optional:

```text
LOG_DIR=/var/log/claude-proxy
OLLAMA_SEARCH_URL=https://ollama.com/api/web_search
SEARCH_MAX_RESULTS=5
SEARCH_SNIPPET_LEN=1500
CODEX_SEARCH_BACKEND=ollama
TAVILY_API_KEY=<optional, used when CODEX_SEARCH_BACKEND=tavily>
TAVILY_SEARCH_URL=https://api.tavily.com/search
```

Do not commit real keys. Use `.env` or your deployment secret store.

## Local Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install fastapi==0.115.0 uvicorn==0.30.6 httpx==0.27.2

export UPSTREAM=http://localhost:4000
export UPSTREAM_API_KEY=...
export OLLAMA_API_KEY=...

uvicorn claude_proxy:app --host 0.0.0.0 --port 4001 --log-level info
```

The legacy entrypoint still works:

```bash
uvicorn proxy:app --host 0.0.0.0 --port 4001
```

## Docker

Build:

```bash
docker build -t gluey-proxy:local .
```

Run:

```bash
docker run --rm -p 4001:4001 \
  -e UPSTREAM=http://host.docker.internal:4000 \
  -e UPSTREAM_API_KEY="$UPSTREAM_API_KEY" \
  -e OLLAMA_API_KEY="$OLLAMA_API_KEY" \
  -e LOG_DIR=/var/log/claude-proxy \
  -v "$PWD/logs:/var/log/claude-proxy" \
  gluey-proxy:local
```

## Production Notes

Production currently uses the image name `claude-proxy:codex-mixed-test` and
mounts logs at `/var/log/claude-proxy`. Test uses the same image with a separate
log directory and test upstream.

Before changing production:

1. Apply the change to the test proxy.
2. Verify Claude Code `/v1/messages` WebSearch.
3. Verify Codex `/v1/responses` without tools, with web search, and with MCP.
4. Promote the exact same image/code to production.

Request logs may contain full prompts and tool outputs. Keep retention short and
do not upload logs without redaction.
