import logging
import re
from typing import List
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


def parse_document_docling(path: str) -> List[str]:
    """Parse document using Docling and return flatten list of strings.
    
    This matches the specific interface required by the existing logic:
    returns a list of 'OCR values' which are meaningful text chunks from the doc.
    """
    logger.info(f"Processing {path} with Docling...")
    
    # Initialize converter
    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document
    
    values: List[str] = []
    seen = set()

    def add_val(text):
        if not text:
            return
        # Normalize
        t = str(text).replace('\n', ' ').replace('\r', ' ').strip()
        t = ' '.join(t.split())
        if len(t) > 1 and t.lower() not in seen:
            values.append(t)
            seen.add(t.lower())

    # Method 1: Export to markdown and parse lines
    # This is the most reliable way to get all text from Docling
    try:
        md_text = doc.export_to_markdown()
        logger.info(f"Docling markdown export length: {len(md_text)} chars")
        
        # Split by lines and clean up markdown syntax
        for line in md_text.split('\n'):
            # Remove markdown formatting
            clean = line.strip()
            # Remove heading markers
            clean = re.sub(r'^#+\s*', '', clean)
            # Remove bold/italic
            clean = re.sub(r'\*+', '', clean)
            # Remove table separators
            if re.match(r'^[\|\-\s:]+$', clean):
                continue
            # Remove pure pipe lines (table structure)
            if clean.startswith('|') and clean.endswith('|'):
                # Extract table cell contents
                cells = [c.strip() for c in clean.split('|') if c.strip()]
                for cell in cells:
                    add_val(cell)
                # Also add combined row
                if len(cells) >= 2:
                    add_val(' '.join(cells))
            else:
                add_val(clean)
    except Exception as e:
        logger.warning(f"Markdown export failed: {e}, trying alternative methods")
    
    # Method 2: Also extract from document text_content if available
    try:
        if hasattr(doc, 'text_content') and doc.text_content:
            for line in str(doc.text_content).split('\n'):
                add_val(line.strip())
    except Exception as e:
        logger.debug(f"text_content extraction failed: {e}")
    
    # Method 3: Iterate over document items
    try:
        for item, _ in doc.iterate_items():
            # Try multiple ways to get text
            if hasattr(item, 'text') and item.text:
                add_val(item.text)
            if hasattr(item, 'content') and item.content:
                add_val(str(item.content))
            # For text items specifically
            if hasattr(item, 'orig') and item.orig:
                add_val(str(item.orig))
    except Exception as e:
        logger.debug(f"iterate_items extraction failed: {e}")

    # Method 4: Handle Tables explicitly using export_to_dataframe
    try:
        for table in doc.tables:
            try:
                df = table.export_to_dataframe(doc=doc)
            except TypeError:
                # Fallback for older API without doc param
                df = table.export_to_dataframe()
            
            # Add column headers
            for col in df.columns:
                add_val(str(col))
                
            # Add cells and combined rows
            for _, row in df.iterrows():
                row_values = []
                for cell in row:
                    val = str(cell) if cell is not None else ""
                    if val.strip():
                        add_val(val)
                        row_values.append(val.strip())
                
                # Combine row for context (important for key:value extraction logic)
                if len(row_values) >= 2:
                    combined = " ".join(row_values)
                    add_val(combined)
    except Exception as e:
        logger.debug(f"Table extraction failed: {e}")

    logger.info(f"Docling extracted {len(values)} unique values")
    
    # Debug: log first few values
    if values:
        logger.info(f"First 10 values: {values[:10]}")
    
    return values


def parse_document_by_page_docling(path: str) -> List[List[str]]:
    """Parse document and return values grouped by page (Docling version)."""
    
    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document
    
    pages_map = {}  # page_no -> list of values
    seen_per_page = {}
    
    def add_to_page(p_no, text):
        if not text:
            return
        t = str(text).replace('\n', ' ').strip()
        t = ' '.join(t.split())
        if len(t) > 1:
            if p_no not in pages_map:
                pages_map[p_no] = []
                seen_per_page[p_no] = set()
            
            if t.lower() not in seen_per_page[p_no]:
                pages_map[p_no].append(t)
                seen_per_page[p_no].add(t.lower())

    # Try to get page-level text from document pages
    try:
        # Docling documents have pages with content
        if hasattr(doc, 'pages') and doc.pages:
            for page_no, page in enumerate(doc.pages, start=1):
                # Try to get text from page
                if hasattr(page, 'text') and page.text:
                    for line in str(page.text).split('\n'):
                        add_to_page(page_no, line.strip())
    except Exception as e:
        logger.debug(f"Page-level extraction failed: {e}")

    # Iterate items with provenance
    try:
        for item, _ in doc.iterate_items():
            page_no = 1
            if hasattr(item, "prov") and item.prov:
                try:
                    page_no = item.prov[0].page_no
                except (IndexError, AttributeError):
                    pass
            
            if hasattr(item, "text") and item.text:
                add_to_page(page_no, item.text)
            if hasattr(item, "content") and item.content:
                add_to_page(page_no, str(item.content))
    except Exception as e:
        logger.debug(f"iterate_items by page failed: {e}")

    # Tables with page info
    try:
        for table in doc.tables:
            page_no = 1
            if hasattr(table, "prov") and table.prov:
                try:
                    page_no = table.prov[0].page_no
                except (IndexError, AttributeError):
                    pass
            
            try:
                df = table.export_to_dataframe(doc=doc)
            except TypeError:
                df = table.export_to_dataframe()
            
            for col in df.columns:
                add_to_page(page_no, str(col))
            
            for _, row in df.iterrows():
                row_vals = []
                for cell in row:
                    val = str(cell) if cell is not None else ""
                    if val.strip():
                        add_to_page(page_no, val)
                        row_vals.append(val.strip())
                if len(row_vals) >= 2:
                    add_to_page(page_no, " ".join(row_vals))
    except Exception as e:
        logger.debug(f"Table by page extraction failed: {e}")

    # If we didn't get page-specific data, raise error to trigger fallback to DocAI
    # We DO NOT want to merge everything into a single page as that breaks multi-invoice PDFs
    if not pages_map:
        raise ValueError("Docling failed to extract page-level data (pages_map empty)")

    # Convert mapping to list of lists, sorted by page number
    sorted_pages = sorted(pages_map.keys())
    result_pages = [pages_map[p] for p in sorted_pages]
    
    logger.info(f"Docling extracted {len(result_pages)} pages with values")
    return result_pages
