.PHONY: help install bot tui dashboard tailwind tailwind-watch build run-bot run-tui run-dashboard logs shell clean

help:
	@echo "chatterbot commands:"
	@echo "  make install         - uv sync (local dev)"
	@echo "  make bot             - run the silent listener locally"
	@echo "  make tui             - run the streamer-only Textual viewer"
	@echo "  make dashboard       - run the FastAPI dashboard"
	@echo "  make tailwind        - build the Tailwind CSS bundle (optional; CDN by default)"
	@echo "  make tailwind-watch  - rebuild Tailwind on file changes"
	@echo "  make build           - build Docker image"
	@echo "  make run-bot         - run bot in Docker"
	@echo "  make run-tui         - run TUI in Docker (interactive)"
	@echo "  make run-dashboard   - run dashboard in Docker"
	@echo "  make logs            - tail container logs"
	@echo "  make shell           - bash shell in container"
	@echo "  make clean           - remove Docker resources"

install:
	uv sync

bot:
	uv run chatterbot bot

tui:
	uv run chatterbot tui

dashboard:
	uv run chatterbot dashboard

tailwind:
	npx --yes tailwindcss -i src/chatterbot/web/static/css/input.css -o src/chatterbot/web/static/css/output.css --minify

tailwind-watch:
	npx --yes tailwindcss -i src/chatterbot/web/static/css/input.css -o src/chatterbot/web/static/css/output.css --watch

build:
	docker compose build

run-bot:
	docker compose run --rm -e RUN_MODE=bot chatterbot

run-tui:
	docker compose run --rm -e RUN_MODE=tui chatterbot

run-dashboard:
	docker compose run --rm --service-ports -e RUN_MODE=dashboard chatterbot

logs:
	docker compose logs -f

shell:
	docker compose run --rm chatterbot /bin/bash

clean:
	docker compose down --rmi local -v
