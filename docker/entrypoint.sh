#!/bin/sh
set -e

# Ollama warms up in the background. The API must not wait for it: /health has
# to be ready inside 60s, and the local model is only the last-resort link in
# the provider chain anyway.
if echo "${LLM_PROVIDER_ORDER:-}" | grep -q ollama; then
  ollama serve >/tmp/ollama.log 2>&1 &

  # Load the weights into memory before the first real request. Cold-loading a
  # 4B model costs ~55s while a warm call costs ~2.5s, and the judge times a
  # request out at 30s — so the first scenario must not be the one that pays it.
  (
    for _ in $(seq 1 60); do
      curl -sf "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null && break
      sleep 1
    done
    curl -sf "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/generate" \
      -d "{\"model\":\"${OLLAMA_MODEL}\",\"prompt\":\"ok\",\"stream\":false,\"keep_alive\":\"24h\"}" \
      >/dev/null 2>&1
  ) &
fi

exec uvicorn gridwise.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-2}" \
  --timeout-keep-alive 65
