# NLP pipeline

This is the canonical home of ValveVie's deterministic clinical-note pipeline.

## Structure

- `pipeline.py` — end-to-end orchestration
- `temporal.py` — implant timeline and pre/post-implant attribution
- `extract_core.py` — clinical finding extraction
- `negation.py` and `sectioning.py` — text preprocessing
- `valve_dictionary.py` — valve model and size recognition
- `varc3_rules.py` — VARC-3/Capodanno staging
- `reviewer_sheet.py` — physician review workbook generator
- `01_notes_nlp_ground_truth.ipynb` — public proof-of-concept notebook
- `notes_nlp_pipeline.ipynb` — working notebook

The Streamlit application imports `run_pipeline` from this package. From the
repository root:

```python
from notebooks.nlp import run_pipeline

labels, audit = run_pipeline("notes_deidentified.xlsx")
```

Run the rule-engine tests with:

```bash
python -m pytest tests/test_varc3_rules.py
```
