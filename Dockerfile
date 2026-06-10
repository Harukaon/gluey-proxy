FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.30.6 httpx==0.27.2
COPY claude_proxy.py /app/claude_proxy.py
ENV LOG_DIR=/var/log/claude-proxy
EXPOSE 4001
CMD ["uvicorn", "claude_proxy:app", "--host", "0.0.0.0", "--port", "4001", "--log-level", "info"]
