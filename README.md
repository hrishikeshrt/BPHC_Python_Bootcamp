# BPHC Python Bootcamp (Sessions 3 & 4)

Hands-on material for the advanced half of the BPHC Python bootcamp. Session 3 focuses on strengthening day-to-day research scripting, while Session 4 layers databases, APIs, parallel processing, and dashboards on top of the same datasets.

## Prerequisites
- Python 3.10+
- LaTeX toolchain (e.g., TeX Live or MacTeX) if you want to rebuild the session notes

## Install Dependencies
```
pip install -r requirements.txt
```
The requirements file covers every Python script in both sessions. Optional packages (for example `psutil` for system stats) are also listed so demos run without extra setup.

## Makefile Shortcuts
Once dependencies are installed, the included `Makefile` provides a few convenience targets:

```
make data      # Generate engineering_test_data.csv and sample experiment folders
make session3  # Run the core Session 3 workflow (data -> batch -> analyzers)
make session4  # Run the core Session 4 workflow (database -> API -> integration demo)
make pdfs      # Rebuild session3.pdf and session4.pdf via pdflatex
make clean     # Remove LaTeX artifacts, __pycache__ folders, and *.db files
```

Run the individual Python scripts directly (as shown below); the Make targets simply prep the shared assets and printable notes.

## Session 3 Workflow (Practical Python Programming)
- **Fast path (2 commands)**
  ```
  python session3/01_create_test_data.py
  python session3/05_advanced_analyzer.py -i engineering_test_data.csv --stats --trends --plot --export-csv --output advanced_results
  ```
- **Full hands-on path**
  1. Warm-up (optional): `python session3/00_basics.py --workspace .`
  2. Seed data: `python session3/01_create_test_data.py`
  3. File operations tour: `python session3/02_file_operations.py`
  4. Batch processing: `python session3/03_batch_processor.py --input-dir experiments --output-dir batch_results --summary --verbose`
  5. Quick stats: `python session3/04_basic_analyzer.py engineering_test_data.csv --stats --verbose`
  6. Deep analysis + plots: `python session3/05_advanced_analyzer.py -i engineering_test_data.csv --stats --trends --plot --export-csv --output advanced_results`

## Session 4 Workflow (Python Integration with Other Technologies)
- **Fast path (single command)**
  ```
  python session4/05_integration_demo.py --output integrated_output
  ```
  Generates a dataset, writes CSV + SQLite, mocks an API, runs parallel stats, and saves plots plus `integrated_system_report.json`.
- **Deep dive (pick and choose)**
  1. SQLite + SQLAlchemy: `python session4/01_database.py`
  2. Mock APIs + analysis: `python session4/02_api_integration.py`
  3. Parallel simulations: `python session4/03_parallel_processing.py`
  4. Flask dashboard backend + plots: `python session4/04_research_dashboard.py`
     - Run server: `flask --app session4/04_research_dashboard.py run --port 5000`
  5. Integration demo (simplified): `python session4/05_integration_demo.py`

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
- Re-run `session3/01_create_test_data.py` any time you want a clean dataset.
- Delete the SQLite files (`research_data.db`, `integrated_research.db`) if you need a fresh database run.
- When using the Flask dashboard, make sure the virtual environment is active so Flask picks up the correct dependencies.

## Further Learning (Self-Study)
- Python language and standard library: https://docs.python.org/3/
- Introductory Python tutorial: https://www.w3schools.com/python/
- Practical articles and recipes: https://realpython.com/
- pandas (data analysis): https://pandas.pydata.org/docs/
- SQLAlchemy (databases): https://docs.sqlalchemy.org/
- Flask (web apps): https://flask.palletsprojects.com/

Happy coding!
