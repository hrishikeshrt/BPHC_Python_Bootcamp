# BPHC Python Bootcamp (Sessions 3 & 4)

Hands-on material for the advanced half of the BPHC Python bootcamp. Session 3 focuses on strengthening day-to-day research scripting, while Session 4 layers databases, APIs, parallel processing, and dashboards on top of the same datasets.

## Prerequisites
- Python 3.10+
- Recommended: `python -m venv .venv && source .venv/bin/activate`
- LaTeX toolchain (e.g., TeX Live or MacTeX) if you want to rebuild the session notes

## Install Dependencies
```
pip install -r requirements.txt
```
The requirements file covers every Python script in both sessions. Optional packages (for example `psutil` for system stats) are also listed so demos run without extra setup.

## Makefile Shortcuts
Once dependencies are installed, the included `Makefile` provides a few convenience targets:

```
make data   # Generate engineering_test_data.csv and sample experiment folders
make pdfs   # Rebuild session3.pdf and session4.pdf via pdflatex
make clean  # Remove LaTeX artifacts, __pycache__ folders, and *.db files
```

Run the individual Python scripts directly during the hands-on session (as shown below); the Make targets simply prep the shared assets and printable notes.

## Session 3 Workflow (Practical Python Programming)
1. **Seed the workspace**
   ```
   python session3/00_create_test_data.py
   ```
   This creates `engineering_test_data.csv` plus an `experiments/` folder that the other exercises consume.
2. **Explore file operations** -- run snippets from `session3/01_file_operations.py` inside a REPL or notebook to contrast manual CSV handling with pandas and JSON metadata.
3. **Batch processing CLI** -- execute:
   ```
   python session3/02_batch_processor.py --input-dir experiments --output-dir batch_results --summary --verbose
   ```
4. **Analyze a single experiment** -- compare the two analyzers:
   ```
   python session3/03_basic_analyzer.py engineering_test_data.csv --stats --verbose
   python session3/04_advanced_analyzer.py -i engineering_test_data.csv --stats --trends --plot --export-csv
   ```

## Session 4 Workflow (Python Integration with Other Technologies)
1. **Persist data in SQLite**
   ```
   python session4/01_database.py
   ```
   The script first looks for `engineering_test_data.csv` produced during Session 3 before generating synthetic data.
2. **Interact with APIs** -- run:
   ```
   python session4/02_api_integration.py
   ```
   This produces `weather_analysis_data.csv` for later experiments.
3. **Parallel simulations**
   ```
   python session4/03_parallel_processing.py
   ```
   Compare sequential vs parallel runtimes and note the `psutil` reminder if the package is missing.
4. **Research dashboard** -- either call the demo mode or start the Flask app:
   ```
   python session4/04_research_dashboard.py
   # or
   flask --app session4/04_research_dashboard run --port 5000
   ```
5. **Full integration demo**
   ```
   python session4/05_integration_demo.py
   ```
   Stores API payloads, triggers parallel jobs, and writes `integrated_system_report.json`.

## Reference Documentation
- `session3.tex` and `session4.tex` mirror the scripts above and can be compiled with `pdflatex` (or your preferred LaTeX workflow).
- `session3.pdf` / `session4.pdf` are pre-built copies for quick sharing.

## Repository Layout
```
BPHC_Python_Bootcamp/
|-- session3/   # Python files for Session 3 (01-05)
|-- session4/   # Python files for Session 4 (01-05)
|-- session3.tex / session3.pdf
|-- session4.tex / session4.pdf
`-- README.md, requirements.txt
```

## Troubleshooting
- Re-run `00_create_test_data.py` any time you want a clean dataset.
- Delete the SQLite files (`research_data.db`, `integrated_research.db`) if you need a fresh database run.
- When using the Flask dashboard, make sure the virtual environment is active so Flask picks up the correct dependencies.

Happy coding!
