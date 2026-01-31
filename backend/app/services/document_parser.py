# backend/app/services/document_parser.py
import os
import mimetypes
import logging
from google.cloud import documentai_v1 as documentai

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def extract_text(doc, anchor):
    if anchor is None:
        return ""
    text = ""
    # defensive: anchor may not have text_segments
    segments = getattr(anchor, "text_segments", None)
    if not segments:
        return ""
    for seg in segments:
        # use start_index/end_index (new API)
        start = getattr(seg, "start_index", None)
        end = getattr(seg, "end_index", None)
        if start is None or end is None:
            continue
        text += doc.text[start:end]
    return text.strip()

def parse_document(file_path, project_id="636776350532", location="us", processor_id="ac3490fa842041e5"):
    client = None
    try:
        client = documentai.DocumentProcessorServiceClient()
    except Exception as e:
        logger.exception("Failed to create Document AI client: %s", e)
        raise

    name = client.processor_path(project_id, location, processor_id)
    logger.info("Using processor: %s", name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "rb") as f:
        file_data = f.read()

    mime_type = mimetypes.guess_type(file_path)[0]
    if mime_type is None:
        mime_type = "application/pdf" if file_path.lower().endswith(".pdf") else "image/jpeg"

    logger.info("Detected MIME type: %s", mime_type)

    raw_doc = documentai.RawDocument(content=file_data, mime_type=mime_type)

    try:
        request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
        result = client.process_document(request=request)
    except Exception as e:
        logger.exception("Document AI process_document failed: %s", e)
        # raise a user-friendly exception with original message
        raise RuntimeError(f"Document AI request failed: {e}") from e

    doc = result.document

    # defensive: ensure doc has text
    doc_text = getattr(doc, "text", "")
    logger.info("Document text length: %d", len(doc_text) if doc_text else 0)

    # extract form fields
    form_fields = {}
    pages = getattr(doc, "pages", [])
    for page in pages:
        for field in getattr(page, "form_fields", []):
            key = extract_text(doc, getattr(field, "field_name", None).text_anchor if getattr(field, "field_name", None) else None)
            val = extract_text(doc, getattr(field, "field_value", None).text_anchor if getattr(field, "field_value", None) else None)
            if key:
                form_fields[key.replace(":", "").strip()] = val.strip()

    # paragraphs
    paragraphs = []
    for page in pages:
        for para in getattr(page, "paragraphs", []):
            text = extract_text(doc, getattr(para, "layout", None).text_anchor if getattr(para, "layout", None) else None)
            if text:
                paragraphs.append(text)

    # tables
    tables = []
    for page in pages:
        for table in getattr(page, "tables", []):
            rows = []
            for row in getattr(table, "body_rows", []):
                cells = []
                for cell in getattr(row, "cells", []):
                    cell_text = extract_text(doc, getattr(cell, "layout", None).text_anchor if getattr(cell, "layout", None) else None)
                    cells.append(cell_text)
                rows.append(cells)
            tables.append(rows)

    return {
        "fields": form_fields,
        "paragraphs": paragraphs,
        "tables": tables,
        "full_text": doc_text
    }

def save_to_txt(result_json, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== FORM FIELDS =====\n")
        for k, v in result_json.get("fields", {}).items():
            f.write(f"{k}: {v}\n")
        f.write("\n\n===== PARAGRAPHS =====\n")
        for p in result_json.get("paragraphs", []):
            f.write(p + "\n")
        f.write("\n\n===== TABLES =====\n")
        for idx, t in enumerate(result_json.get("tables", [])):
            f.write(f"\n--- Table {idx+1} ---\n")
            for row in t:
                f.write(" | ".join(row) + "\n")
        f.write("\n\n===== FULL OCR TEXT =====\n")
        f.write(result_json.get("full_text", ""))
