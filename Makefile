PYTHON ?= python3
LATEX ?= pdflatex

.PHONY: data pdfs clean

# Generate shared datasets used across both sessions
data:
	$(PYTHON) session3/00_create_test_data.py

# Rebuild the printable notes
pdfs:
	$(LATEX) session3.tex >/dev/null
	$(LATEX) session4.tex >/dev/null

# Remove build artifacts and scratch files
clean:
	rm -f session3.{aux,log,out} session4.{aux,log,out}
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -f *.db
