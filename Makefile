# The Same Barrel — one entry point for everything.
#
#   make install    install pinned dependencies
#   make test       run the test suite
#   make verify     check the headline numbers against the committed series (no API key)
#   make smoke      synthetic-data run of the engine (no API key)
#   make numbers    regenerate every number in the pitch      (needs price data)
#   make derived    regenerate the committed derived CSVs     (needs price data)
#   make exhibits   regenerate the figures                    (needs price data)
#   make all        numbers + derived + exhibits

export PYTHONPATH := $(CURDIR):$(CURDIR)/engine:$(CURDIR)/data/fetch

PRICES ?= data/px_clean.parquet

.PHONY: install test verify smoke numbers derived exhibits all clean

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

exhibits:
	@mkdir -p docs/figures
	python exhibits/make_exhibits.py --prices $(PRICES)
	python exhibits/riskfigs.py --prices $(PRICES)

all: numbers derived exhibits

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
