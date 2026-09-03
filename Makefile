# The Same Barrel — one entry point for everything.
#
#   make install    install pinned dependencies
#   make test       run the test suite
#   make verify     check the headline numbers against the committed series (no API key)
#   make smoke      synthetic-data run of the engine (no API key)
#   make numbers    regenerate every number in the pitch      (needs price data)
#   make derived    regenerate the committed derived CSVs     (needs price data)
#   make figures    regenerate all figures from derived data (no API key)
#   make exhibits   regenerate the research exhibits         (needs price data)
#   make all        numbers + derived + figures + exhibits

export PYTHONPATH := $(CURDIR):$(CURDIR)/engine:$(CURDIR)/data/fetch

PRICES ?= data/px_clean.parquet

.PHONY: install test verify smoke numbers derived figures exhibits all clean

install:
	pip install -r requirements.txt

test:
	pytest tests/

verify:
	python verify_from_derived.py

smoke:
	python engine/immediacy.py --smoke

numbers:
	@mkdir -p docs
	python engine/final_numbers.py --prices $(PRICES) > docs/FINAL_NUMBERS.txt
	@echo "wrote docs/FINAL_NUMBERS.txt"

derived:
	python data/export_derived.py --prices $(PRICES)

figures:
	@mkdir -p docs/figures
	python exhibits/concept_figure.py
	python exhibits/readme_figures.py
	python exhibits/pitch_exhibit_page.py

all: numbers derived figures exhibits

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
