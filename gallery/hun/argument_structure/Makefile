# ============================================================================
# Argument Structure Pipeline - Makefile (Hungarian)
# ============================================================================
#
# Quick Start:
#   make help              - Show available targets
#   make data              - Quick test run (limited verbs)
#   make data-full         - Production run (all verbs, all pairs)
#   make test              - Run the test suite
#   make show-stats        - Show statistics about generated data
#   make clean             - Clean generated items and cache
#
# ============================================================================

PROJECT_ROOT := $(shell cd ../../.. && pwd)
PYTHON := uv run --project $(PROJECT_ROOT) python

LEXICONS_DIR     := lexicons
TEMPLATES_DIR    := templates
FILLED_DIR       := filled_templates
ITEMS_DIR        := items
LISTS_DIR        := lists
RESOURCES_DIR    := resources
CACHE_DIR        := .cache

# Default limits for quick/test runs
DEFAULT_VERB_COUNT         := 100
DEFAULT_2AFC_LIMIT         := 50
DEFAULT_PER_TEMPLATE_LIMIT := 3

# Constituent order for generated frames: preverbal (Stage-1 default) or
# neutral. See generate_templates.py and LINGUISTIC_DESIGN_NOTES.md section 7.
WORD_ORDER := preverbal

BLUE   := \033[0;34m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m

# ============================================================================
# Main Targets
# ============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Argument Structure Pipeline (Hungarian) - Available Targets$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Main Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(all|data|help|show-stats|show-config)"
	@echo ""
	@echo "$(GREEN)Data Generation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(resources|lexicons|templates|fill|2afc|lists|data-full)" | \
		grep -v -E "(deployment|clean)"
	@echo ""
	@echo "$(GREEN)Deployment:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "deployment" | grep -v "clean"
	@echo ""
	@echo "$(GREEN)Quality:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(lint|check|test|validate)"
	@echo ""
	@echo "$(GREEN)Cleaning:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep "clean"
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make data                        # Quick test run"
	@echo "  make data-full                   # Production run (all verbs)"
	@echo "  make templates WORD_ORDER=neutral  # Verb-medial constituent order"
	@echo "  make templates-nominal           # Nominal frames only"
	@echo "  make test                        # Run the test suite"
	@echo "  make samples                     # Print one example per frame"
	@echo ""

.PHONY: all
all: data-full deployment-full ## Run full production pipeline
	@echo "$(GREEN)✓ Full pipeline executed successfully$(NC)"

.PHONY: data
data: resources lexicons templates fill-templates 2afc-pairs lists ## Generate all data files with limits (quick test run)
	@echo "$(GREEN)✓ All data files generated$(NC)"

.PHONY: data-full
data-full: resources lexicons templates fill-templates-full 2afc-pairs-full lists ## Generate all data files without limits (production run)
	@echo "$(GREEN)✓ Full data pipeline executed$(NC)"

# ============================================================================
# Data Generation Targets
# ============================================================================

.PHONY: resources
resources: ## Rebuild generated resource CSVs (bleached noun paradigm)
	@echo "$(BLUE)[0/5] Building generated resources...$(NC)"
	$(PYTHON) $(RESOURCES_DIR)/build_bleached_nouns.py
	@echo "$(GREEN)✓ Resources built$(NC)"

.PHONY: check-resources
check-resources: ## Verify generated CSVs match their generator
	@echo "$(BLUE)Checking generated resources are up to date...$(NC)"
	$(PYTHON) $(RESOURCES_DIR)/build_bleached_nouns.py --check
	@echo "$(GREEN)✓ Resources up to date$(NC)"

.PHONY: lexicons
lexicons: ## Generate lexicons (UniMorph verbs + bleached resources)
	@echo "$(BLUE)[1/5] Generating lexicons...$(NC)"
	@echo "$(YELLOW)⚠ Downloads Hungarian UniMorph on first run$(NC)"
	$(PYTHON) generate_lexicons.py
	@echo "$(GREEN)✓ Lexicons generated$(NC)"

.PHONY: lexicons-quick
lexicons-quick: ## Generate lexicons with a capped verb inventory
	@echo "$(BLUE)[1/5] Generating lexicons (limit=$(DEFAULT_VERB_COUNT) lemmas)...$(NC)"
	$(PYTHON) generate_lexicons.py --limit $(DEFAULT_VERB_COUNT)
	@echo "$(GREEN)✓ Lexicons generated$(NC)"

.PHONY: templates
templates: ## Generate the full frame inventory (WORD_ORDER=$(WORD_ORDER))
	@echo "$(BLUE)[2/5] Generating templates ($(WORD_ORDER) order)...$(NC)"
	$(PYTHON) generate_templates.py --word-order $(WORD_ORDER)
	@echo "$(GREEN)✓ Templates generated$(NC)"

.PHONY: templates-nominal
templates-nominal: ## Generate nominal (case-marked argument) frames only
	@echo "$(BLUE)[2/5] Generating nominal frames only...$(NC)"
	$(PYTHON) generate_templates.py --word-order $(WORD_ORDER) --include nominal
	@echo "$(GREEN)✓ Templates generated$(NC)"

.PHONY: templates-postpositions
templates-postpositions: ## Generate postposition frames only (spatial + complex)
	@echo "$(BLUE)[2/5] Generating postposition frames only...$(NC)"
	$(PYTHON) generate_templates.py --word-order $(WORD_ORDER) --include spatial complex
	@echo "$(GREEN)✓ Templates generated$(NC)"

.PHONY: fill-templates
fill-templates: ## Fill templates (limit=$(DEFAULT_VERB_COUNT) verbs) — quick test run
	@echo "$(BLUE)[3/5] Filling templates ($(DEFAULT_VERB_COUNT) verbs, max $(DEFAULT_PER_TEMPLATE_LIMIT)/template)...$(NC)"
	$(PYTHON) fill_templates.py --verb-count $(DEFAULT_VERB_COUNT) --max-per-template $(DEFAULT_PER_TEMPLATE_LIMIT)
	@echo "$(GREEN)✓ Templates filled$(NC)"

.PHONY: fill-templates-full
fill-templates-full: ## Fill all templates with all verbs — production run
	@echo "$(BLUE)[3/5] Filling all templates (all verbs)...$(NC)"
	@echo "$(YELLOW)⚠ This step may take a while$(NC)"
	$(PYTHON) fill_templates.py
	@echo "$(GREEN)✓ Templates filled$(NC)"

.PHONY: fill-dry-run
fill-dry-run: ## Fill five representative templates with five verbs
	@echo "$(BLUE)Filling representative subset...$(NC)"
	$(PYTHON) fill_templates.py --dry-run
	@echo "$(GREEN)✓ Dry run complete$(NC)"

.PHONY: 2afc-pairs
2afc-pairs: ## Generate 2AFC pairs (limit=$(DEFAULT_2AFC_LIMIT)) — quick test run
	@echo "$(BLUE)[4/5] Generating 2AFC pairs (limit=$(DEFAULT_2AFC_LIMIT))...$(NC)"
	$(PYTHON) create_2afc_pairs.py --limit $(DEFAULT_2AFC_LIMIT)
	@echo "$(GREEN)✓ 2AFC pairs generated$(NC)"

.PHONY: 2afc-pairs-full
2afc-pairs-full: ## Generate 2AFC pairs from all filled templates (no limit)
	@echo "$(BLUE)[4/5] Generating 2AFC pairs (no limit)...$(NC)"
	@echo "$(YELLOW)⚠ This step uses NYTK/PULI-GPT-2 and may take a while$(NC)"
	$(PYTHON) create_2afc_pairs.py
	@echo "$(GREEN)✓ All 2AFC pairs generated$(NC)"

.PHONY: 2afc-pairs-structural
2afc-pairs-structural: ## Generate 2AFC pairs without LM scoring (structural inspection)
	@echo "$(BLUE)Generating 2AFC pairs without scoring...$(NC)"
	@echo "$(YELLOW)⚠ For inspection only; final runs must score$(NC)"
	$(PYTHON) create_2afc_pairs.py --skip-scoring
	@echo "$(GREEN)✓ 2AFC pairs generated (unscored)$(NC)"

.PHONY: lists
lists: ## Partition 2AFC pairs into experiment lists
	@echo "$(BLUE)[5/5] Generating experiment lists...$(NC)"
	$(PYTHON) generate_lists.py
	@echo "$(GREEN)✓ Experiment lists generated$(NC)"

.PHONY: deployment
deployment: ## Generate jsPsych deployment for 2 lists (local testing)
	@echo "$(BLUE)Generating deployment (2 lists)...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 2
	@echo "$(GREEN)✓ Deployment generated$(NC)"

.PHONY: deployment-full
deployment-full: ## Generate deployment for all 16 lists
	@echo "$(BLUE)Generating full deployment (16 lists)...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 16
	@echo "$(GREEN)✓ Full deployment generated$(NC)"

# ============================================================================
# Quality Assurance Targets
# ============================================================================

.PHONY: test
test: ## Run the Hungarian pipeline test suite
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -q
	@echo "$(GREEN)✓ Tests passed$(NC)"

.PHONY: samples
samples: ## Print one rendered example per frame (no UniMorph needed)
	@echo "$(BLUE)Rendering one example per frame ($(WORD_ORDER) order)...$(NC)"
	$(PYTHON) -m tests.render_samples --word-order $(WORD_ORDER) --per-template 1

.PHONY: lint
lint: ## Run ruff linting on all pipeline scripts
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(PYTHON) -m ruff check \
		generate_lexicons.py \
		generate_templates.py \
		fill_templates.py \
		create_2afc_pairs.py \
		generate_lists.py \
		generate_deployment.py \
		resources/build_bleached_nouns.py \
		tests/
	@echo "$(GREEN)✓ Linting checks passed$(NC)"

.PHONY: check
check: check-resources lint test ## Run resource check, linting and tests
	@echo "$(GREEN)✓ All checks passed$(NC)"

.PHONY: validate
validate: ## Build the human linguistic review sheet
	@echo "$(BLUE)Building human review sheet...$(NC)"
	$(PYTHON) validation/build_human_review.py
	@echo "$(GREEN)✓ Review sheet built$(NC)"

# ============================================================================
# Data Inspection Targets
# ============================================================================

.PHONY: show-stats
show-stats: ## Show counts for all generated data files
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Data Statistics$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Resources:$(NC)"
	@for f in $(RESOURCES_DIR)/*.csv; do \
		[ -f "$$f" ] && printf "  %-34s %6s rows\n" "$$(basename $$f)" "$$(($$(wc -l < $$f) - 1))"; \
	done
	@echo ""
	@echo "$(GREEN)Lexicons:$(NC)"
	@if [ -d "$(LEXICONS_DIR)" ]; then \
		for f in $(LEXICONS_DIR)/*.jsonl; do \
			[ -f "$$f" ] && printf "  %-34s %6s entries\n" "$$(basename $$f)" "$$(wc -l < $$f)"; \
		done; \
	else echo "  (not generated)"; fi
	@echo ""
	@echo "$(GREEN)Templates:$(NC)"
	@if [ -f "$(TEMPLATES_DIR)/generic_frames.jsonl" ]; then \
		printf "  %-34s %6s frames\n" "generic_frames.jsonl" "$$(wc -l < $(TEMPLATES_DIR)/generic_frames.jsonl)"; \
	else echo "  (not generated)"; fi
	@echo ""
	@echo "$(GREEN)Filled templates:$(NC)"
	@if [ -d "$(FILLED_DIR)" ]; then \
		for f in $(FILLED_DIR)/*.jsonl; do \
			[ -f "$$f" ] && printf "  %-34s %6s sentences\n" "$$(basename $$f)" "$$(wc -l < $$f)"; \
		done; \
	else echo "  (not generated)"; fi
	@echo ""
	@echo "$(GREEN)Items and lists:$(NC)"
	@if [ -d "$(ITEMS_DIR)" ]; then \
		for f in $(ITEMS_DIR)/*.jsonl; do \
			[ -f "$$f" ] && printf "  %-34s %6s items\n" "$$(basename $$f)" "$$(wc -l < $$f)"; \
		done; \
	fi
	@if [ -d "$(LISTS_DIR)" ]; then \
		for f in $(LISTS_DIR)/*.jsonl; do \
			[ -f "$$f" ] && printf "  %-34s %6s lists\n" "$$(basename $$f)" "$$(wc -l < $$f)"; \
		done; \
	fi
	@echo ""

.PHONY: show-config
show-config: ## Print key configuration values
	@echo "$(BLUE)Key configuration (config.yaml):$(NC)"
	@grep -E "^\s*(language_code|model_name|n_lists|items_per_list|n_quantiles):" config.yaml || true
	@echo ""
	@echo "$(BLUE)Makefile settings:$(NC)"
	@echo "  WORD_ORDER = $(WORD_ORDER)"
	@echo "  DEFAULT_VERB_COUNT = $(DEFAULT_VERB_COUNT)"

.PHONY: show-frames
show-frames: ## List generated frame names and their surface strings
	@$(PYTHON) -c "import json; \
		[print(f\"{d['name']:38s} {d['template_string']}\") \
		 for d in map(json.loads, open('$(TEMPLATES_DIR)/generic_frames.jsonl', encoding='utf-8'))]"

# ============================================================================
# Cleaning Targets
# ============================================================================

.PHONY: clean
clean: ## Remove generated items, lists and model cache
	@echo "$(BLUE)Cleaning generated items and cache...$(NC)"
	rm -rf $(CACHE_DIR)
	rm -f $(ITEMS_DIR)/*.jsonl
	rm -f $(LISTS_DIR)/*.jsonl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

.PHONY: clean-all
clean-all: clean ## Remove all generated files including lexicons and templates
	@echo "$(RED)Removing all generated data...$(NC)"
	rm -f $(LEXICONS_DIR)/*.jsonl
	rm -f $(TEMPLATES_DIR)/*.jsonl
	rm -f $(FILLED_DIR)/*.jsonl
	rm -rf deployment
	@echo "$(GREEN)✓ All generated files removed$(NC)"
