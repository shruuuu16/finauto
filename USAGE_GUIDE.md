# FinAutomate Usage Guide

## 🚀 Starting the Application

1. **Start the FastAPI backend:**
```bash
cd FinAutomate
python main.py
```
Or if you have uvicorn installed:
```bash
uvicorn main:app --reload
```

2. **Access the application:**
Open your browser and go to: `http://127.0.0.1:8000/frontend/index.html`

## 📋 Three Extraction Modes

### 1. 🔍 Extract Only
- **What it does:** Extracts all text from your document
- **Output:** Shows extracted values on screen (no files generated)
- **When to use:** When you just want to see what text the OCR found
- **Requirements:** Only the document file

### 2. 📄 Extract with PDF
- **What it does:** Extracts text and creates a PDF summary
- **Output:** PDF file with all extracted values
- **When to use:** When you want a document with all extracted text
- **Requirements:** Only the document file (no template needed)

### 3. 📊 Extract with Template
- **What it does:** Extracts text, matches it to template fields, fills Excel template
- **Output:** Filled Excel file + PDF summary of matched fields
- **When to use:** When you have a template and want structured data extraction
- **Requirements:** Document file + Excel template file

## 🛠️ How It Works

### Template Mode Details:
1. Upload your invoice/document (PDF or image)
2. Upload your Excel template with field labels
3. The system will:
   - Extract all text using Google Document AI OCR
   - Find matching values for each template field
   - Use AI (Gemini) to select the best match
   - Fill the template with extracted values
   - Generate confidence scores for each field

### Field Matching:
- **High confidence (≥0.5):** Value written directly
- **Low confidence (<0.5):** Marked as `[?] value`
- **No match found:** Marked as `[Not Found]`

## 📝 Tips for Best Results

1. **Template Format:**
   - Put field labels in one column
   - Leave the next column empty for extracted values
   - Example:
     ```
     Invoice Number: | [extracted value goes here]
     Invoice Date:   | [extracted value goes here]
     Total Amount:   | [extracted value goes here]
     ```

2. **Field Labels:**
   - Use clear, descriptive labels
   - Match common invoice terminology
   - Examples: "Invoice Date", "Total Amount", "Invoice Number"

3. **Document Quality:**
   - Use clear, high-resolution scans
   - Ensure text is readable
   - PDFs work better than images for complex documents

## 🔧 Troubleshooting

### Values not filling correctly?
- Check that template labels match the document structure
- Review the confidence scores in the PDF summary
- Ensure document quality is good

### Server not starting?
- Make sure all requirements are installed: `pip install -r requirements.txt`
- Check Google Cloud credentials are set up correctly
- Verify ports 8000 is not in use

### Frontend not loading?
- Ensure you're accessing: `http://127.0.0.1:8000/frontend/index.html`
- Check browser console for errors
- Verify CORS is enabled in main.py

## 📊 Output Files

All generated files are saved in the `output/` directory with unique job IDs:
- `{job_id}_filled.xlsx` - Filled template (Template Mode)
- `{job_id}_summary.pdf` - Extraction summary

## 🎯 Example Workflow

1. Start server: `python main.py`
2. Open browser: `http://127.0.0.1:8000/frontend/index.html`
3. Select your invoice PDF
4. Choose extraction mode:
   - Just want to see extracted text? → **Extract Only**
   - Want a PDF report? → **Extract with PDF**
   - Have a template to fill? → **Extract with Template** (upload template too)
5. Click the button and wait for processing
6. Download your results!

---

Made with ❤️ for automated invoice processing
