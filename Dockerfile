# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Ollama binary and the local model live in their own layers so that editing
# application code never re-downloads several hundred MB.
RUN curl -fsSL https://ollama.com/install.sh | sh

USER app
ENV OLLAMA_MODELS=/home/app/.ollama/models \
    OLLAMA_HOST=http://127.0.0.1:11434

ARG LOCAL_MODEL=qwen2.5:3b-instruct
ENV OLLAMA_MODEL=${LOCAL_MODEL}

# Pulled at BUILD time, never at startup: /health has to answer within 60s of
# the container starting, which a few-hundred-MB download cannot guarantee.
# LOCAL_MODEL=none skips it, for a small image that runs on cloud providers only.
RUN if [ "$OLLAMA_MODEL" != "none" ]; then \
      ollama serve >/tmp/ollama-build.log 2>&1 & \
      for _ in $(seq 1 60); do \
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; \
        sleep 1; \
      done; \
      ollama pull "$OLLAMA_MODEL"; \
    fi

WORKDIR /app
COPY --chown=app:app gridwise ./gridwise
COPY --chown=app:app docker/entrypoint.sh ./entrypoint.sh

ENV PORT=8000 \
    LLM_PROVIDER_ORDER=gemini,groq,ollama

EXPOSE 8000

# No credentials are baked in; keys arrive as environment variables at run time.
ENTRYPOINT ["/bin/sh", "./entrypoint.sh"]
