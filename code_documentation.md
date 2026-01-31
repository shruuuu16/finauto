# Codebase Documentation

This document describes the main extraction scripts in the repository and how to use them.

**Files documented**
- [main.py](main.py) — lightweight Document AI example that runs a processor and prints JSON
- [extraction.py](extraction.py) — helper-style extraction functions and a writer to save unified text output

---

**Overview: main.py**

- Purpose: Call Google Cloud Document AI to process a PDF, extract form fields, paragraphs, and tables, and return a JSON structure.
- Requirements: `google-cloud-documentai` Python package and Google credentials accessible by the environment (for example, `GOOGLE_APPLICATION_CREDENTIALS`).

Functions
- `extract_text(doc, anchor)`
  - Params: `doc` (Document AI document object), `anchor` (text anchor from a layout)
  - Returns: `str` — the selected substring from `doc.text` (handles None)
  - Notes: Joins multiple `text_segments` and strips whitespace

- `parse_document(file_path)`
  - Params: `file_path` (path to a PDF)
  - Returns: `dict` — JSON-like dict with keys: `fields`, `paragraphs`, `tables`, `full_text`
  - Behavior: Creates a `DocumentProcessorServiceClient`, calls the configured processor (hard-coded `project_id`, `location`, `processor_id`), and extracts:
    - `fields`: dict mapping form label → value (collected from `page.form_fields`)
    - `paragraphs`: list of paragraph strings
    - `tables`: list of tables; each table is a list of rows, each row is a list of cell strings
    - `full_text`: raw OCR text from the Document AI `doc.text`
  - Notes: The function currently hardcodes `project_id`, `location`, and `processor_id`; consider moving these to configuration or environment variables.

- `get_field(result_json, field_name)`
  - Params: `result_json` (output from `parse_document`), `field_name` (label to search)
  - Returns: `str` or `None` — best-effort case-insensitive match; exact match first, then partial

Command / Example

- Run directly (example):

```bash
python main.py
```

The script calls `parse_document("sample1.pdf")` when executed as `__main__`. Replace `sample1.pdf` with your file or import `parse_document` from code.

Notes & Recommendations
- Set Google credentials via `GOOGLE_APPLICATION_CREDENTIALS` or an appropriate auth method.
- Remove hard-coded `project_id` / `processor_id` and read from environment or a config file.
- Consider adding robust error handling around the API call and file I/O.

---

**Overview: extraction.py**

- Purpose: Utility functions to parse a PDF with Document AI and a helper to save the unified extraction to a `.txt` file.
- Requirements: same as `main.py` (`google-cloud-documentai` and credentials)

Functions
- `extract_text(doc, anchor)`
  - Same behavior as the function in `main.py` (returns joined text segments or empty string)

- `parse_document(file_path)`
  - Same output schema as `main.parse_document`:
    - `fields`: dict of form fields
    - `paragraphs`: list of paragraph strings
    - `tables`: list of tables (rows of cell strings)
    - `full_text`: entire `doc.text`
  - Implementation differences: more list comprehensions for paragraphs and table parsing (cleaner but equivalent)

- `save_to_txt(result_json, output_path)`
  - Params: `result_json` (output from `parse_document`), `output_path` (file to write)
  - Behavior: Ensures output directory exists and writes a readable text file grouping `FORM FIELDS`, `PARAGRAPHS`, `TABLES`, and `FULL OCR TEXT`.
  - Notes: Uses UTF-8 and simple table row formatting with ` | ` separators.

Command / Example usage

```python
from extraction import parse_document, save_to_txt

result = parse_document("invoice.pdf")
save_to_txt(result, "out/invoice_extraction.txt")
```

Recommendations
- `parse_document` and `save_to_txt` are already reusable utilities — consider exposing them via a small CLI or an API entrypoint.
- Add error handling when writing files and when the Document AI client call fails.

---

**Output schema (both scripts)**

The functions return a dict with this shape:

- `fields`: { "Label": "Value", ... }
- `paragraphs`: [ "Paragraph 1 text", "Paragraph 2", ... ]
- `tables`: [ [ [cell1, cell2, ...], [row2cell1, ...] ], ... ] (list of tables → each table is list of rows → each row is list of cell strings)
- `full_text`: large string of all OCR text

---


---