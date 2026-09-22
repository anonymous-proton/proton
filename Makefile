OPS_COMPOSE_FILE ?= ops/docker-compose.yml
OPS_COMPOSE = docker compose -f $(OPS_COMPOSE_FILE)

.PHONY: ops-up ops-down

ops-up:
	$(OPS_COMPOSE) up -d --build

ops-down:
	$(OPS_COMPOSE) down
