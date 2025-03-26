.DEFAULT_GOAL := help

HOST ?= 0.0.0.0
PORT ?= 8123

run:  ## Run the application using uvicorn with provided arguments or defaults
	@echo "----- Run uvicorn-----"
	uv run uvicorn main:app --host $(HOST) --port $(PORT) --reload

install:  ## Install a dependency using uv
	@echo "----- Installing dependency $(LIBRARY) -----"
	uv add $(LIBRARY)

uninstall:  ## Uninstall a dependency using uv
	@echo "----- Uninstalling dependency $(LIBRARY) -----"
	uv remove $(LIBRARY)

help:  ## Show help message
	@echo "----- Usage: make [command] -----"
	@echo ""
	@echo "Commands: "
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
