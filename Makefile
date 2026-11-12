.DEFAULT_GOAL := help

.PHONY: help install scan dry-run clean-state

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install/sync project dependencies via uv
	uv sync

scan: install ## Scan all receipts in inputs/ (ARGS="--sleep 5" to pass extra flags)
	uv run scan.py $(ARGS)

dry-run: install ## List receipts that would be scanned, no API calls
	uv run scan.py --dry-run $(ARGS)

clean-state: ## Remove cached OCR text and quota tracking in outputs/ (keeps receipts.csv)
	rm -rf outputs/ocr outputs/.quota.json
