.PHONY: run build test clean init help

CARGO := cargo
CONFIG ?= config.json

help:
	@echo "Codex Convert Proxy - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make run CONFIG=config.json   Run the proxy with config file"
	@echo "  make init                    Generate config.example.json"
	@echo "  make build                  Build the project"
	@echo "  make test                   Run tests"
	@echo "  make clean                  Clean build artifacts"
	@echo ""

run:
	$(CARGO) run -- start --config $(CONFIG)

init:
	$(CARGO) run -- init config.example.json

build:
	$(CARGO) build

test:
	$(CARGO) test

clean:
	$(CARGO) clean

# Development shortcuts
dev:
	$(CARGO) build
	$(CARGO) run -- start --provider glm --upstream-url https://api.example.com --api-key test --listen 0.0.0.0:8080

check:
	$(CARGO) check
	$(CARGO) clippy -- -D warnings
