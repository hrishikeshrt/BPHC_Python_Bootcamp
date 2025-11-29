PYTHON ?= python3
LATEX ?= pdflatex

.PHONY: data pdfs clean session3 session4

# Generate shared datasets used across both sessions
data:
	$(PYTHON) session3/01_create_test_data.py

# Run the core Session 3 workflow
session3:
	$(PYTHON) session3/01_create_test_data.py
	$(PYTHON) session3/03_batch_processor.py --input-dir experiments --output-dir batch_results --summary
	$(PYTHON) session3/04_basic_analyzer.py engineering_test_data.csv --stats
	$(PYTHON) session3/05_advanced_analyzer.py -i engineering_test_data.csv --stats --trends --plot --export-csv --output advanced_results

# Run the core Session 4 workflow
session4:
	$(PYTHON) session4/01_database.py
	$(PYTHON) session4/02_api_integration.py
	$(PYTHON) session4/05_integration_demo.py --output integrated_output

# Rebuild the printable notes
pdfs:
	$(LATEX) session3.tex >/dev/null
	$(LATEX) session4.tex >/dev/null

# Remove build artifacts and scratch files
clean:
	rm -f session3.{aux,log,out} session4.{aux,log,out}
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -f *.db
