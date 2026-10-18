# ============================================================================
# Argument Structure Pipeline - Makefile (Korean)
# ============================================================================
#
# Quick Start:
#   make help              - Show available targets
#   make data              - Quick test run (small scoring sample, 2 lists)
#   make data-full         - Production run (250 per cell, 16 lists)
#   make deployment        - jsPsych/JATOS deployment for 2 lists (local test)
#   make simulate          - Active-learning simulation (klue/roberta-base)
#   make show-stats        - Show statistics about generated data
#
# Pipeline: lexicons -> frames -> score concrete templates -> bin/select ->
#           2AFC pairs -> lists -> deployment. Parameters live in config.yaml.
#
# ============================================================================

# Python interpreter (use uv run with project root for dependency management)
PROJECT_ROOT := $(shell cd ../../.. && pwd)
PYTHON := uv run --project $(PROJECT_ROOT) python

# Directories
LEXICONS_DIR     := lexicons
TEMPLATES_DIR    := templates
SCORES_DIR       := concrete_template_scores
FILLED_DIR       := filled_templates
ITEMS_DIR        := items
LISTS_DIR        := lists
CACHE_DIR        := .cache

# Quick-run scoring sample per frame_class x aspect cell (production value:
# config.yaml concrete_template_scoring.sample_per_cell)
QUICK_SAMPLE_PER_CELL := 40

# Past-tense test path: number of lists (must match config.yaml past_test.n_lists)
PAST_N_LISTS := 27

# Legacy exhaustive-fill path limits
DEFAULT_FILL_LIMIT := 100
DEFAULT_2AFC_LIMIT := 50
DEFAULT_PER_TEMPLATE_LIMIT := 100

# Colors for output
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
	@echo "$(BLUE)Argument Structure Pipeline (Korean) - Available Targets$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Main Targets:$(NC)"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-26s$(NC) %s\n", $$1, $$2}' | \
		grep -E "^  .{0,12}(all|data|data-full|help|show-stats|show-config|validate-protocol) "
	@echo ""
	@echo "$(GREEN)Pipeline Steps:$(NC)"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-26s$(NC) %s\n", $$1, $$2}' | \
		grep -E "\[[0-9]/7\]"
	@echo ""
	@echo "$(GREEN)Legacy exhaustive-fill path (superseded):$(NC)"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-26s$(NC) %s\n", $$1, $$2}' | \
		grep "Legacy"
	@echo ""
	@echo "$(GREEN)Quality and simulation:$(NC)"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-26s$(NC) %s\n", $$1, $$2}' | \
		grep -E "^  .{0,12}(lint|check|simulate)"
	@echo ""
	@echo "$(GREEN)Cleaning:$(NC)"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-26s$(NC) %s\n", $$1, $$2}' | \
		grep -E "^  .{0,12}clean"
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make data                   # Quick test run (40 per cell, 2 lists)"
	@echo "  make data-full              # Production run (250 per cell, 16 lists)"
	@echo "  make deployment             # Deployment for 2 lists (local test)"
	@echo "  make deployment-full        # Deployment for all 16 lists"
	@echo "  make show-stats             # Counts for all generated files"
	@echo ""

.PHONY: all
all: data-full deployment-full ## Run the full production pipeline including deployment of all 16 lists
	@echo "$(GREEN)✓ Full pipeline executed successfully$(NC)"

.PHONY: data
data: validate-protocol lexicons templates-full score-concrete-templates-quick bin-concrete-templates concrete-template-pairs lists ## Quick test run: small scoring sample (40 per cell), 2 lists
	@echo "$(GREEN)✓ All data files generated (quick)$(NC)"

.PHONY: data-full
data-full: validate-protocol lexicons templates-full score-concrete-templates bin-concrete-templates concrete-template-pairs lists-full ## Production run: 250 per cell, 16 lists
	@echo "$(GREEN)✓ Full data pipeline executed$(NC)"

# ============================================================================
# Pipeline Steps
# ============================================================================

.PHONY: data-past
data-past: validate-protocol lexicons templates-full sample-concrete-templates select-pairs lists-past ## Past-tense test: per-frame sample, carrier score, pairs binned by score gap, 27 lists
	@echo "$(GREEN)✓ Past-tense test data generated$(NC)"

.PHONY: validate-protocol
validate-protocol: ## Validate the protocol section of config.yaml builds cleanly (prompt, drift checks)
	@echo "$(BLUE)Validating protocol declaration...$(NC)"
	@$(PYTHON) -c "from protocol import build_protocol; \
		p = build_protocol('config.yaml'); \
		assert p.families, 'No families declared in protocol'; \
		anchor = p.families[0].anchor; \
		print('  protocol:', p.name); \
		print('  family:', anchor.name); \
		print('  prompt:', anchor.canonical_prompt); \
		print('  scale_type:', anchor.response_space.scale_type)"
	@$(PYTHON) -c "from protocol import check_canonical_prompt; \
		r = check_canonical_prompt('config.yaml'); \
		print('  drift checks on prompt:', 'passed' if r.passed else 'FAILED ' + str(r.findings)); \
		raise SystemExit(0 if r.passed else 1)"
	@echo "$(GREEN)✓ Protocol validated$(NC)"

.PHONY: lexicons
lexicons: ## [1/7] Generate lexicons (UniMorph verbs + bleached resources)
	@echo "$(BLUE)[1/7] Generating lexicons...$(NC)"
	$(PYTHON) generate_lexicons.py
	@echo "$(GREEN)✓ Lexicons generated$(NC)"

.PHONY: templates-full
templates-full: ## [2/7] Generate all 138 frames (base, adjunct, source-goal, spatial, complex, complement, progressive)
	@echo "$(BLUE)[2/7] Generating all frames...$(NC)"
	$(PYTHON) generate_templates.py
	@echo "$(GREEN)✓ Frames generated$(NC)"

.PHONY: templates
templates: ## [2/7] Generate base frames only (intransitive + transitive) — smoke test
	@echo "$(BLUE)[2/7] Generating base frames...$(NC)"
	$(PYTHON) generate_templates.py --include
	@echo "$(GREEN)✓ Frames generated$(NC)"

.PHONY: score-concrete-templates
score-concrete-templates: ## [3/7] Score concrete templates, per-cell sample (config: concrete_template_scoring)
	@echo "$(BLUE)[3/7] Scoring concrete templates (config sample per cell)...$(NC)"
	$(PYTHON) score_concrete_templates.py
	@echo "$(GREEN)✓ Concrete templates scored$(NC)"

.PHONY: score-concrete-templates-quick
score-concrete-templates-quick: ## [3/7] Score a small concrete-template sample (40 per cell) — quick test
	@echo "$(BLUE)[3/7] Scoring concrete templates ($(QUICK_SAMPLE_PER_CELL) per cell)...$(NC)"
	$(PYTHON) score_concrete_templates.py --sample-per-cell $(QUICK_SAMPLE_PER_CELL)
	@echo "$(GREEN)✓ Concrete templates scored$(NC)"

.PHONY: bin-concrete-templates
bin-concrete-templates: ## [4/7] Bin scores per frame_class x aspect cell and select per bin (config: concrete_template_binning)
	@echo "$(BLUE)[4/7] Binning and selecting concrete templates...$(NC)"
	$(PYTHON) bin_concrete_templates.py
	@echo "$(GREEN)✓ Concrete templates selected$(NC)"

.PHONY: concrete-template-pairs
concrete-template-pairs: ## [5/7] Build 2AFC pairs from selected concrete templates (config: concrete_template_pairs)
	@echo "$(BLUE)[5/7] Building 2AFC pairs...$(NC)"
	$(PYTHON) create_concrete_template_pairs.py
	@echo "$(GREEN)✓ Pairs built -> $(ITEMS_DIR)/2afc_pairs.jsonl$(NC)"

.PHONY: sample-concrete-templates
sample-concrete-templates: ## [past] Sample concrete templates per simple frame + two-mask carrier score (config: past_test)
	@echo "$(BLUE)[past] Sampling and scoring concrete templates...$(NC)"
	$(PYTHON) sample_concrete_templates.py

.PHONY: select-pairs
select-pairs: ## [past] Build, score, bin (by score gap) and select past-tense 2AFC pairs (config: past_test)
	@echo "$(BLUE)[past] Building and selecting 2AFC pairs...$(NC)"
	$(PYTHON) select_pairs.py
	@echo "$(GREEN)✓ Pairs selected -> $(ITEMS_DIR)/2afc_pairs.jsonl$(NC)"

.PHONY: lists-past
lists-past: ## [past] Partition the past-tense pairs into PAST_N_LISTS (27) lists of 25 + 25
	@echo "$(BLUE)[past] Generating $(PAST_N_LISTS) experiment lists...$(NC)"
	$(PYTHON) generate_lists.py --n-lists $(PAST_N_LISTS)
	@echo "$(GREEN)✓ Experiment lists generated$(NC)"

.PHONY: lists
lists: ## [6/7] Partition pairs into 2 lists — quick test
	@echo "$(BLUE)[6/7] Generating 2 experiment lists...$(NC)"
	$(PYTHON) generate_lists.py --n-lists 2
	@echo "$(GREEN)✓ Experiment lists generated$(NC)"

.PHONY: lists-full
lists-full: ## [6/7] Partition pairs into config n_lists (16) lists of 25 + 25
	@echo "$(BLUE)[6/7] Generating experiment lists (config n_lists)...$(NC)"
	$(PYTHON) generate_lists.py
	@echo "$(GREEN)✓ Experiment lists generated$(NC)"

.PHONY: deployment
deployment: ## [7/7] jsPsych/JATOS deployment for 2 lists (local test)
	@echo "$(BLUE)[7/7] Generating deployment (2 lists, local + jatos)...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 2
	@echo "$(GREEN)✓ Deployment generated$(NC)"
	@echo "  Open: deployment/local/list_01/index.html"

.PHONY: deployment-full
deployment-full: ## [7/7] jsPsych/JATOS deployment for all 16 lists
	@echo "$(BLUE)[7/7] Generating full deployment (16 lists)...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 16
	@echo "$(GREEN)✓ Full deployment generated$(NC)"

.PHONY: deployment-local-only
deployment-local-only: ## [7/7] Local-only deployment for all 16 lists (no JATOS .jzip export)
	@echo "$(BLUE)[7/7] Generating local-only deployment...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 16 --no-jatos
	@echo "$(GREEN)✓ Local deployment generated$(NC)"

# ============================================================================
# Active-Learning Simulation
# ============================================================================

.PHONY: simulate
simulate: ## Simulate the active-learning loop (config: simulation; klue/roberta-base, uncertainty sampling)
	@echo "$(BLUE)Simulating active learning (uncertainty sampling)...$(NC)"
	$(PYTHON) simulate_pipeline.py --selector uncertainty
	@echo "$(GREEN)✓ Simulation done -> simulation_output/$(NC)"

.PHONY: simulate-baseline
simulate-baseline: ## Same simulation with random selection, for comparison
	@echo "$(BLUE)Simulating active learning (random selection baseline)...$(NC)"
	$(PYTHON) simulate_pipeline.py --selector random
	@echo "$(GREEN)✓ Simulation done -> simulation_output/$(NC)"

.PHONY: simulate-quick
simulate-quick: ## 2-round, 1-epoch simulation — checks the loop runs (~30 s on MPS)
	$(PYTHON) simulate_pipeline.py --max-iterations 2 --epochs 1 --output simulation_output/simulation_quick.json

# ============================================================================
# Legacy exhaustive-fill path (superseded by concrete-template sampling;
# kept for reference — the full cross-product is ~14.5 billion sentences)
# ============================================================================

.PHONY: fill-templates
fill-templates: ## Legacy: fill frames exhaustively (100 verbs, max 100 per frame)
	@echo "$(BLUE)Legacy: filling frames (limit=$(DEFAULT_FILL_LIMIT) verbs, max $(DEFAULT_PER_TEMPLATE_LIMIT)/frame)...$(NC)"
	$(PYTHON) fill_templates.py --limit $(DEFAULT_FILL_LIMIT) --max-per-template $(DEFAULT_PER_TEMPLATE_LIMIT)
	@echo "$(GREEN)✓ Frames filled$(NC)"

.PHONY: fill-templates-full
fill-templates-full: ## Legacy: fill all frames with all verbs (very slow)
	@echo "$(BLUE)Legacy: filling all frames with all verbs...$(NC)"
	@echo "$(YELLOW)⚠ This step may take a very long time$(NC)"
	$(PYTHON) fill_templates.py
	@echo "$(GREEN)✓ Frames filled$(NC)"

.PHONY: 2afc-pairs
2afc-pairs: ## Legacy: 2AFC pairs from 50 filled frames; overwrites items/2afc_pairs.jsonl
	@echo "$(BLUE)Legacy: generating 2AFC pairs (limit=$(DEFAULT_2AFC_LIMIT))...$(NC)"
	$(PYTHON) create_2afc_pairs.py --limit $(DEFAULT_2AFC_LIMIT)
	@echo "$(GREEN)✓ 2AFC pairs generated$(NC)"

.PHONY: 2afc-pairs-full
2afc-pairs-full: ## Legacy: 2AFC pairs from all filled frames; overwrites items/2afc_pairs.jsonl
	@echo "$(BLUE)Legacy: generating 2AFC pairs (no limit)...$(NC)"
	@echo "$(YELLOW)⚠ This step uses EleutherAI/polyglot-ko-1.3b and may take a while$(NC)"
	$(PYTHON) create_2afc_pairs.py
	@echo "$(GREEN)✓ All 2AFC pairs generated$(NC)"

# ============================================================================
# Quality Assurance Targets
# ============================================================================

.PHONY: lint
lint: ## Run ruff linting on all pipeline scripts
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(PYTHON) -m ruff check \
		protocol.py \
		generate_lexicons.py \
		generate_templates.py \
		score_concrete_templates.py \
		bin_concrete_templates.py \
		create_concrete_template_pairs.py \
		sample_concrete_templates.py \
		select_pairs.py \
		generate_lists.py \
		generate_deployment.py \
		simulate_pipeline.py
	@echo "$(GREEN)✓ Linting checks passed$(NC)"

.PHONY: check
check: lint ## Run static analysis
	@echo "$(GREEN)✓ Static analysis passed$(NC)"

# ============================================================================
# Data Inspection Targets
# ============================================================================

.PHONY: show-stats
show-stats: ## Show counts and sizes for all generated data files
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Data Statistics$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Lexicons:$(NC)"
	@if [ -d "$(LEXICONS_DIR)" ]; then \
		for file in $(LEXICONS_DIR)/*.jsonl; do \
			if [ -f "$$file" ]; then \
				count=$$(wc -l < "$$file"); \
				size=$$(ls -lh "$$file" | awk '{print $$5}'); \
				echo "  $$(basename $$file): $$count entries ($$size)"; \
			fi \
		done; \
	else \
		echo "  $(RED)No lexicons found. Run 'make lexicons' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Frames:$(NC)"
	@if [ -f "$(TEMPLATES_DIR)/generic_frames.jsonl" ]; then \
		count=$$(wc -l < "$(TEMPLATES_DIR)/generic_frames.jsonl"); \
		size=$$(ls -lh "$(TEMPLATES_DIR)/generic_frames.jsonl" | awk '{print $$5}'); \
		echo "  generic_frames.jsonl: $$count frames ($$size)"; \
	else \
		echo "  $(RED)No frames found. Run 'make templates-full' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Concrete templates:$(NC)"
	@if [ -d "$(SCORES_DIR)" ]; then \
		for file in $(SCORES_DIR)/*.jsonl; do \
			if [ -f "$$file" ]; then \
				count=$$(wc -l < "$$file"); \
				size=$$(ls -lh "$$file" | awk '{print $$5}'); \
				echo "  $$(basename $$file): $$count rows ($$size)"; \
			fi \
		done; \
	else \
		echo "  $(RED)No scores found. Run 'make score-concrete-templates' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Items:$(NC)"
	@if [ -f "$(ITEMS_DIR)/2afc_pairs.jsonl" ]; then \
		count=$$(wc -l < "$(ITEMS_DIR)/2afc_pairs.jsonl"); \
		size=$$(ls -lh "$(ITEMS_DIR)/2afc_pairs.jsonl" | awk '{print $$5}'); \
		echo "  2afc_pairs.jsonl: $$count pairs ($$size)"; \
	else \
		echo "  $(RED)2afc_pairs.jsonl not found. Run 'make concrete-template-pairs' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Lists:$(NC)"
	@if [ -f "$(LISTS_DIR)/experiment_lists.jsonl" ]; then \
		count=$$(wc -l < "$(LISTS_DIR)/experiment_lists.jsonl"); \
		size=$$(ls -lh "$(LISTS_DIR)/experiment_lists.jsonl" | awk '{print $$5}'); \
		echo "  experiment_lists.jsonl: $$count lists ($$size)"; \
	else \
		echo "  $(RED)No lists found. Run 'make lists' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Deployment:$(NC)"
	@if [ -d "deployment/local" ]; then \
		n=$$(ls -d deployment/local/list_* 2>/dev/null | wc -l); \
		echo "  local: $$n lists"; \
	else \
		echo "  $(RED)No local deployment found. Run 'make deployment' first.$(NC)"; \
	fi
	@if [ -d "deployment/jatos" ]; then \
		n=$$(ls deployment/jatos/*.jzip 2>/dev/null | wc -l); \
		echo "  jatos: $$n .jzip packages"; \
	else \
		echo "  $(RED)No jatos deployment found.$(NC)"; \
	fi
	@echo ""

.PHONY: show-config
show-config: ## Show key pipeline configuration values
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Pipeline Configuration$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@$(PYTHON) -c "import yaml; c = yaml.safe_load(open('config.yaml')); \
		s, b, p, l, e = c['concrete_template_scoring'], c['concrete_template_binning'], \
			c['concrete_template_pairs'], c['lists'], c['deployment']['experiment']; \
		print('Project:', c['project']['name'], '/', c['project']['language_code']); \
		print(''); \
		print('Scoring sample per cell:', s['sample_per_cell'], '(seed', str(s['seed']) + ')'); \
		print('Bins per cell:', b['n_bins'], '| selected per bin:', b['per_bin']); \
		print('Pairs per concrete template:', p['different_verb_per_concrete_template'], 'different_verb +', \
			p['same_verb_per_concrete_template'], 'same_verb | drop_adjectives:', p['drop_adjectives']); \
		print(''); \
		print('Lists:', l['n_lists'], 'x', l['items_per_list'], '| split:', l['split_by']['per_list']); \
		print('Quantile bins (lm_score_diff):', l['quantile_bins']); \
		print(''); \
		print('Deployment title:', e['title'], '| language:', e['language'])"
	@echo ""

# ============================================================================
# Cleaning Targets
# ============================================================================

.PHONY: clean
clean: clean-items clean-cache ## Clean generated items and model cache
	@echo "$(GREEN)✓ Cleaned$(NC)"

.PHONY: clean-items
clean-items: ## Clean 2AFC pairs (and legacy cross-product items)
	@echo "$(BLUE)Cleaning generated items...$(NC)"
	@rm -f $(ITEMS_DIR)/cross_product_items.jsonl
	@rm -f $(ITEMS_DIR)/2afc_pairs.jsonl $(ITEMS_DIR)/2afc_pairs_scoring.json
	@echo "$(GREEN)✓ Items cleaned$(NC)"

.PHONY: clean-concrete-templates
clean-concrete-templates: ## Clean concrete-template scores, selections and bin summary
	@echo "$(BLUE)Cleaning concrete-template scores...$(NC)"
	@rm -f $(SCORES_DIR)/*.jsonl $(SCORES_DIR)/*.json
	@echo "$(GREEN)✓ Concrete-template scores cleaned$(NC)"

.PHONY: clean-cache
clean-cache: ## Clean MLM/LM model cache
	@echo "$(BLUE)Cleaning model cache...$(NC)"
	@rm -rf $(CACHE_DIR)
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

.PHONY: clean-filled-templates
clean-filled-templates: ## Clean legacy filled frames
	@echo "$(BLUE)Cleaning filled frames...$(NC)"
	@rm -f $(FILLED_DIR)/*.jsonl
	@echo "$(GREEN)✓ Filled frames cleaned$(NC)"

.PHONY: clean-templates
clean-templates: ## Clean generated frames
	@echo "$(BLUE)Cleaning frames...$(NC)"
	@rm -f $(TEMPLATES_DIR)/generic_frames.jsonl
	@echo "$(GREEN)✓ Frames cleaned$(NC)"

.PHONY: clean-lexicons
clean-lexicons: ## Clean generated lexicons
	@echo "$(BLUE)Cleaning lexicons...$(NC)"
	@rm -f $(LEXICONS_DIR)/*.jsonl
	@echo "$(GREEN)✓ Lexicons cleaned$(NC)"

.PHONY: clean-lists
clean-lists: ## Clean experiment lists
	@echo "$(BLUE)Cleaning experiment lists...$(NC)"
	@rm -f $(LISTS_DIR)/*.jsonl
	@echo "$(GREEN)✓ Lists cleaned$(NC)"

.PHONY: clean-deployment
clean-deployment: ## Clean deployment files
	@echo "$(BLUE)Cleaning deployment files...$(NC)"
	@rm -rf deployment/local deployment/jatos
	@echo "$(GREEN)✓ Deployment cleaned$(NC)"

.PHONY: clean-all
.PHONY: clean-simulation
clean-simulation: ## Clean simulation results
	@rm -rf simulation_output
	@echo "$(GREEN)✓ Simulation results cleaned$(NC)"

clean-all: clean-simulation clean-items clean-concrete-templates clean-cache clean-filled-templates clean-templates clean-lexicons clean-lists clean-deployment ## Clean all generated files
	@echo "$(RED)⚠ All generated data removed$(NC)"

# ============================================================================
# Special Targets
# ============================================================================

.DEFAULT_GOAL := help
.SECONDARY:
.PHONY: all data data-full help clean check lint
