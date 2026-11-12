PYTHON ?= python3

TESTS := \
	recovery/tests/test_metrics.py \
	recovery/tests/test_fall_handover.py \
	recovery/tests/test_robustness_stats.py \
	pilots/g4_humanlike_getup_20261108/test_selection.py

.PHONY: help test check-links

help:
	@echo "Targets:"
	@echo "  make test          run the CPU unit tests (torch/numpy only, no Isaac Lab)"
	@echo "  make check-links   check relative links and images in the tracked Markdown files"
	@echo "  make help          show this message"
	@echo ""
	@echo "Variables: PYTHON=<interpreter> (default: python3). Example:"
	@echo "  make test PYTHON=/path/to/isaac-lab/bin/python"

test:
	@set -e; for t in $(TESTS); do echo "== $$t"; $(PYTHON) -B -I "$$t"; done; echo "all CPU tests passed"

check-links:
	$(PYTHON) tools/check_md_links.py
