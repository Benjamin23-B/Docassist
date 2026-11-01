.PHONY: env up down restart logs logs-worker logs-frontend ps migrate revision shell test lint format

env:
	@test -f .env || cp .env.example .env

up: env
	docker compose up -d --build

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f api

logs-worker:
	docker compose logs -f worker

logs-frontend:
	docker compose logs -f frontend

ps:
	docker compose ps

migrate:
	docker compose exec api alembic upgrade head

revision:
	docker compose exec api alembic revision --autogenerate -m "$(m)"

shell:
	docker compose exec api python

test:
	cd backend && python -m pytest

lint:
	cd backend && ruff check .

format:
	cd backend && ruff format .
