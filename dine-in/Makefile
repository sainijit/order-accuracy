.PHONY: help build up down logs restart benchmark benchmark-stop benchmark-status clean

help:
	@echo "Dine-In Order Accuracy - Available Commands:"
	@echo ""
	@echo "  make build          - Build Docker images"
	@echo "  make up             - Start services (UI mode)"
	@echo "  make up-benchmark   - Start services with benchmark mode enabled"
	@echo "  make down           - Stop all services"
	@echo "  make logs           - View application logs"
	@echo "  make restart        - Restart dine-in service"
	@echo "  make benchmark      - Start benchmark mode via API"
	@echo "  make benchmark-stop - Stop benchmark mode"
	@echo "  make benchmark-status - Get benchmark status"
	@echo "  make clean          - Remove containers and volumes"
	@echo ""

build:
	@echo "Building dine-in Docker images..."
	docker compose build

up:
	@echo "Starting dine-in services (UI mode)..."
	docker compose up -d
	@echo "Services started:"
	@echo "  - Gradio UI: http://localhost:7861"
	@echo "  - API: http://localhost:8083"
	@echo "  - API Docs: http://localhost:8083/docs"

up-benchmark:
	@echo "Starting dine-in services (Benchmark mode)..."
	BENCHMARK_MODE=true docker compose up -d
	@echo "Services started in benchmark mode"
	@echo "  - API: http://localhost:8083"
	@echo "  - Start benchmark: make benchmark"

down:
	@echo "Stopping dine-in services..."
	docker compose down

logs:
	@echo "Showing dine-in application logs (Ctrl+C to exit)..."
	docker logs -f dinein_app

restart:
	@echo "Restarting dine-in service..."
	docker compose restart dine-in

benchmark:
	@echo "Starting benchmark mode..."
	@curl -X POST http://localhost:8083/api/benchmark/start \
		-H "Content-Type: application/json" | jq
	@echo ""
	@echo "Benchmark started. Monitor with: make benchmark-status"

benchmark-stop:
	@echo "Stopping benchmark mode..."
	@curl -X POST http://localhost:8083/api/benchmark/stop \
		-H "Content-Type: application/json" | jq

benchmark-status:
	@curl -s http://localhost:8083/api/benchmark/status | jq

clean:
	@echo "Cleaning up containers and volumes..."
	docker compose down -v
	@echo "Cleanup complete"

# Development helpers
shell:
	@echo "Opening shell in dine-in container..."
	docker exec -it dinein_app /bin/bash

test-api:
	@echo "Testing API health..."
	@curl -s http://localhost:8083/health | jq
