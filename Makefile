.PHONY: help install bot tui dashboard all tailwind tailwind-watch build run-bot run-tui run-dashboard logs shell clean

help:
	@echo "chatterbot commands:"
	@echo "  make install         - uv sync (local dev)"
	@echo "  make bot             - run the silent listener locally"
	@echo "  make tui             - run the streamer-only Textual viewer"
	@echo "  make dashboard       - run the FastAPI dashboard"
	@echo "  make all             - bot in background + dashboard in foreground (Ctrl+C kills both)"
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

# Run bot in the background (logs to logs/bot.log, auto-restart on exit) plus
# dashboard in the foreground. Ctrl+C in the dashboard shell kills both.
#
# The bot is wrapped in a fail-fast restart loop: a clean exit (e.g. SIGTERM
# from the dashboard's "Restart bot" button) is followed by a fresh launch
# within ~2s. A near-immediate crash (<5s, e.g. bad token) breaks the loop
# so a config bug doesn't hot-spin the host.
all:
	@mkdir -p logs
	@( while true; do \
	    START=$$(date +%s); \
	    uv run chatterbot bot >> logs/bot.log 2>&1; \
	    END=$$(date +%s); \
	    if [ $$((END - START)) -lt 5 ]; then \
	      echo "bot exited in <5s — likely a config error. check logs/bot.log" >> logs/bot.log; \
	      break; \
	    fi; \
	    echo "bot exited; restarting in 2s..." >> logs/bot.log; \
	    sleep 2; \
	  done ) & \
	LOOP_PID=$$!; \
	echo "bot launched in restart-loop (logs/bot.log) — tail with: tail -f logs/bot.log"; \
	echo "dashboard starting in foreground (Ctrl+C stops both)..."; \
	trap "echo; echo 'stopping bot loop pid $$LOOP_PID'; \
	      pkill -P $$LOOP_PID 2>/dev/null; \
	      kill $$LOOP_PID 2>/dev/null" INT TERM EXIT; \
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
