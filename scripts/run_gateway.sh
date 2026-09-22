#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS_COMPOSE_FILE="${REPO_ROOT}/ops/docker-compose.yml"
LOG_DIR="${REPO_ROOT}/monitoring/runtime/logs"
COLLECTOR_DIR="${REPO_ROOT}/monitoring/runtime/collector"
GATEWAY_LOG_FILE="${LOG_DIR}/gateway.log"
COLLECTOR_LOG_FILE="${LOG_DIR}/collector.log"

gateway_host="127.0.0.1"
gateway_http_host=""
gateway_http_port="8098"

args=("$@")
has_no_stream_logs="0"
for ((i = 0; i < ${#args[@]}; i++)); do
  arg="${args[$i]}"
  case "${arg}" in
    --host=*)
      gateway_host="${arg#*=}"
      ;;
    --host)
      if (( i + 1 < ${#args[@]} )); then
        gateway_host="${args[$((i + 1))]}"
      fi
      ;;
    --http-host=*)
      gateway_http_host="${arg#*=}"
      ;;
    --http-host)
      if (( i + 1 < ${#args[@]} )); then
        gateway_http_host="${args[$((i + 1))]}"
      fi
      ;;
    --http-port=*)
      gateway_http_port="${arg#*=}"
      ;;
    --http-port)
      if (( i + 1 < ${#args[@]} )); then
        gateway_http_port="${args[$((i + 1))]}"
      fi
      ;;
    --no-stream-logs|--no-stream-logs=*)
      has_no_stream_logs="1"
      ;;
  esac
done

if [[ -z "${gateway_http_host}" ]]; then
  gateway_http_host="${gateway_host}"
fi

if [[ "${has_no_stream_logs}" != "1" ]]; then
  args+=("--no-stream-logs")
fi

gateway_endpoint="http://${gateway_http_host}:${gateway_http_port}"
collector_pid=""
compose_started="0"

mkdir -p "${LOG_DIR}"
mkdir -p "${COLLECTOR_DIR}"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${collector_pid}" ]] && kill -0 "${collector_pid}" 2>/dev/null; then
    kill "${collector_pid}" 2>/dev/null || true
    wait "${collector_pid}" 2>/dev/null || true
  fi

  if [[ "${compose_started}" == "1" ]]; then
    docker compose -f "${OPS_COMPOSE_FILE}" down --remove-orphans >/dev/null 2>&1 || true
  fi

  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

{
  echo "[$(date -Iseconds)] starting gateway orchestration"
  echo "[$(date -Iseconds)] observability compose: ${OPS_COMPOSE_FILE}"
  echo "[$(date -Iseconds)] collector endpoint: ${gateway_endpoint}"
  echo "[$(date -Iseconds)] worker logs are collected via Loki docker scrape"
  echo "[$(date -Iseconds)] command: python -m gateway ${args[*]}"
} >> "${GATEWAY_LOG_FILE}"

cd "${REPO_ROOT}"
GATEWAY_BASE_URL="${gateway_endpoint}" docker compose -f "${OPS_COMPOSE_FILE}" up -d --remove-orphans
compose_started="1"

python scripts/collect_observability.py \
  --config configs/monitoring.yaml \
  --gateway-endpoint "${gateway_endpoint}" \
  --heartbeat-s 30 \
  >> "${COLLECTOR_LOG_FILE}" 2>&1 &
collector_pid=$!

sleep 1
if ! kill -0 "${collector_pid}" 2>/dev/null; then
  {
    echo "[$(date -Iseconds)] WARNING: collector exited unexpectedly; gateway will continue"
  } >> "${GATEWAY_LOG_FILE}"
  collector_pid=""
fi

python -m gateway "${args[@]}" 2>&1 | tee -a "${GATEWAY_LOG_FILE}" | sed -u '/^INFO:aiohttp.access:/d'
