console.log("NEW JavaScript loaded!");

let currentJobId = null;
let pollInterval = null;
let activePollJobId = null;
let isBusy = false;
// Store accumulated files
let accumulatedFiles = [];
// API_BASE empty for relative paths (Vercel)
const API_BASE = "";
// Elements
const debugBar = document.getElementById("debugBar");
const docInput = document.getElementById("document");
const tplInput = document.getElementById("template");
const extractBtn = document.getElementById("extractBtn");
const extractPdfBtn = document.getElementById("extractPdfBtn");
const extractTemplateBtn = document.getElementById("extractTemplateBtn");
const jobIdEl = document.getElementById("jobId");
const statusEl = document.getElementById("status");
const extractedValuesDiv = document.getElementById("extractedValues");
const valuesListEl = document.getElementById("valuesList");
const resultDiv = document.getElementById("result");
const excelLink = document.getElementById("excelLink");
const pdfLink = document.getElementById("pdfLink");

console.log("Elements:", {
    debugBar,
    docInput,
    tplInput,
    extractBtn,
    extractPdfBtn,
    extractTemplateBtn,
    jobIdEl,
    statusEl,
    resultDiv,
    excelLink,
    pdfLink,
});

const docName = document.getElementById("docName");
const tplName = document.getElementById("tplName");

function updateFileDisplay() {
    if (accumulatedFiles.length === 0) {
        docName.textContent = "";
    } else if (accumulatedFiles.length === 1) {
        docName.textContent = accumulatedFiles[0].name;
    } else {
        docName.textContent = `${accumulatedFiles.length} files selected`;
    }
}

// Accumulate files when selected (instead of replacing)
docInput.addEventListener("change", (e) => {
    const files = e.target.files;
    for (let i = 0; i < files.length; i++) {
        // Avoid duplicates by name
        const exists = accumulatedFiles.some(f => f.name === files[i].name);
        if (!exists) {
            accumulatedFiles.push(files[i]);
        }
    }
    updateFileDisplay();
    setDebug(`Added ${files.length} file(s). Total: ${accumulatedFiles.length}`);
});

tplInput.addEventListener("change", (e) => {
    tplName.textContent = e.target.files[0]?.name || "";
});

// Clear accumulated files function
function clearFiles() {
    accumulatedFiles = [];
    docInput.value = "";
    updateFileDisplay();
    setDebug("Files cleared");
}

// Wire up clear button
const clearFilesBtn = document.getElementById("clearFilesBtn");
if (clearFilesBtn) {
    clearFilesBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearFiles();
    });
}

function setDebug(message) {
    if (debugBar) debugBar.textContent = `Debug: ${message}`;
    console.log("[UI]", message);
}

function setStatus(text, color) {
    statusEl.textContent = text;
    statusEl.style.fontWeight = "bold";
    statusEl.style.color = color || "#334155";
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    activePollJobId = null;
}

function setBusy(busy) {
    isBusy = busy;
    const disabled = Boolean(busy);
    if (extractBtn) extractBtn.disabled = disabled;
    if (extractPdfBtn) extractPdfBtn.disabled = disabled;
    if (extractTemplateBtn) extractTemplateBtn.disabled = disabled;
}

function resetUiForNewJob() {
    extractedValuesDiv.style.display = "none";
    valuesListEl.innerHTML = "";
    if (resultDiv) resultDiv.style.display = "none";
    if (excelLink) excelLink.style.display = "none";
    if (pdfLink) pdfLink.style.display = "none";
}

function toPublicUrl(pathStr) {
    if (!pathStr || typeof pathStr !== "string") return null;
    const normalized = pathStr.replace(/\\/g, "/");
    if (normalized.startsWith("http://") || normalized.startsWith("https://")) return normalized;
    return `${API_BASE}/${normalized.replace(/^\//, "")}`;
}

async function startJob({ endpoint, formData, modeName }) {
    if (isBusy) {
        setDebug("Busy - wait for current job");
        return;
    }

    setBusy(true);
    stopPolling();
    resetUiForNewJob();
    jobIdEl.textContent = "Waiting...";
    setStatus("Uploading...", "#2563eb");
    setDebug(`${modeName}: uploading`);

    try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: "POST",
            body: formData,
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            setDebug(`${modeName}: upload failed (${res.status})`);
            setStatus("Upload failed", "#dc2626");
            setBusy(false);
            alert("Upload failed. Check backend terminal or logs.");
            return;
        }

        // VERCEL SYNC MODE SUPPORT
        if (data.status === "done") {
            setDebug(`${modeName}: sync job done immediately`);
            setStatus("done", "#16a34a");
            // Handle results directly
            handleJobDone(data, modeName);
            return;
        }

        currentJobId = data.job_id;
        activePollJobId = currentJobId;
        jobIdEl.textContent = currentJobId || "(missing job_id)";
        setStatus("processing", "#f59e0b");
        setDebug(`${modeName}: job created ${currentJobId}`);

        // Immediate status check then poll
        await checkStatus(currentJobId, modeName);
        pollInterval = setInterval(() => checkStatus(currentJobId, modeName), 1000);
    } catch (error) {
        setDebug(`${modeName}: network error: ${error?.message || String(error)}`);
        setStatus("Network error", "#dc2626");
        setBusy(false);
    }
}

async function handleExtractOnly() {
    const docFile = docInput?.files?.[0];
    if (!docFile) {
        setDebug("No file selected");
        alert("Please select a document first!");
        return;
    }

    const formData = new FormData();
    formData.append("document", docFile);
    await startJob({ endpoint: "/extract-only", formData, modeName: "Extract Only" });
}

async function handleExtractWithPdf() {
    const docFile = docInput?.files?.[0];
    if (!docFile) {
        setDebug("No file selected");
        alert("Please select a document first!");
        return;
    }

    const formData = new FormData();
    formData.append("document", docFile);
    await startJob({ endpoint: "/upload", formData, modeName: "Extract with PDF" });
}

async function handleExtractWithTemplate() {
    const tplFile = tplInput?.files?.[0];

    if (accumulatedFiles.length === 0) {
        setDebug("No document selected");
        alert("Please select at least one document first!");
        return;
    }

    if (!tplFile) {
        setDebug("No template selected");
        alert("Please select an Excel template for this mode!");
        return;
    }

    const formData = new FormData();

    // Check if multiple documents
    const sheetVal = document.getElementById("sheetCategory")?.value;

    if (accumulatedFiles.length > 1) {
        // Multiple documents - use upload-multi endpoint
        for (let i = 0; i < accumulatedFiles.length; i++) {
            formData.append("documents", accumulatedFiles[i]);
        }
        formData.append("template", tplFile);
        if (sheetVal) formData.append("sheet_name", sheetVal);

        setDebug(`Uploading ${accumulatedFiles.length} documents with template (Sheet: ${sheetVal || "Auto"})...`);
        await startJob({ endpoint: "/upload-multi", formData, modeName: `Extract ${accumulatedFiles.length} Documents` });
    } else {
        // Single document - use regular upload endpoint
        formData.append("document", accumulatedFiles[0]);
        formData.append("template", tplFile);
        if (sheetVal) formData.append("sheet_name", sheetVal);

        setDebug(`Uploading document with template (Sheet: ${sheetVal || "Auto"})...`);
        await startJob({ endpoint: "/upload", formData, modeName: "Extract with Template" });
    }

    // Clear accumulated files after upload
    accumulatedFiles = [];
    docInput.value = "";
    updateFileDisplay();
}

async function checkStatus(jobId, modeName) {
    if (!jobId) return;
    if (activePollJobId && activePollJobId !== jobId) return;

    try {
        const res = await fetch(`${API_BASE}/status/${jobId}`);
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            // In Vercel, status endpoint might not find the job if it wasn't valid, but let's assume it works for async.
            setDebug(`Status check failed (${res.status})`);
            return;
        }

        if (data.status) {
            setStatus(data.status, data.status === "done" ? "#16a34a" : data.status === "failed" ? "#dc2626" : "#f59e0b");
        }

        if (data.status === "done") {
            stopPolling();
            setBusy(false);
            handleJobDone(data, modeName);
        }

        if (data.status === "failed") {
            stopPolling();
            setBusy(false);
            setDebug(`${modeName}: failed: ${data.error || "unknown"}`);
        }
    } catch (error) {
        setDebug(`${modeName}: status network error: ${error?.message || String(error)}`);
    }
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function handleJobDone(data, modeName) {
    // Extract-only values
    const values = Array.isArray(data.ocr_values) ? data.ocr_values : [];
    if (values.length > 0) {
        extractedValuesDiv.style.display = "block";
        valuesListEl.innerHTML = "";
        values.forEach((value, idx) => {
            const p = document.createElement("p");
            p.style.padding = "8px";
            p.style.margin = "5px 0";
            p.style.background = "#f8fafc";
            p.style.borderLeft = "3px solid #2563eb";
            p.textContent = `${idx + 1}. ${value}`;
            valuesListEl.appendChild(p);
        });
        setDebug(`${modeName}: rendered ${values.length} values`);
    }

    // PDF/Excel links
    // Handle both URL (server path) and Base64 (sync mode)
    let pdfUrl = toPublicUrl(data.pdf);
    let excelUrl = toPublicUrl(data.excel);

    if (data.pdf_base64) {
        const blob = base64ToBlob(data.pdf_base64, "application/pdf");
        pdfUrl = URL.createObjectURL(blob);
    }
    if (data.excel_base64) {
        const blob = base64ToBlob(data.excel_base64, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        excelUrl = URL.createObjectURL(blob);
    }

    console.log("Job data:", data); // Debug

    if (pdfUrl || excelUrl) {
        if (resultDiv) resultDiv.style.display = "block";
        if (pdfLink && pdfUrl) {
            pdfLink.href = pdfUrl;
            pdfLink.style.display = "inline-block";
            // If it's a blob, maybe set download attribute?
            if (pdfUrl.startsWith("blob:")) {
                pdfLink.download = "summary.pdf";
            } else {
                pdfLink.removeAttribute("download");
            }
        }
        if (excelLink && excelUrl) {
            excelLink.href = excelUrl;
            excelLink.style.display = "inline-block";
            if (excelUrl.startsWith("blob:")) {
                excelLink.download = "result.xlsx";
            } else {
                excelLink.removeAttribute("download");
            }
        }

        // Show sheet selection info if available
        if (data.selected_sheet && resultDiv) {
            const sheetInfo = document.createElement("div");
            sheetInfo.style.cssText = "padding: 12px; margin: 10px 0; background: #dbeafe; border-left: 4px solid #2563eb; border-radius: 4px; font-size: 14px;";
            sheetInfo.id = "sheet-info-display";

            let infoHTML = `<strong>📋 Selected Sheet:</strong> <span style="color: #1e40af; font-weight: 600;">${data.selected_sheet}</span>`;
            if (data.document_type) {
                infoHTML += ` <span style="color: #4b5563; font-size: 0.9em;">(${data.document_type.replace('_', ' ')})</span>`;
            }
            if (data.available_sheets && data.available_sheets.length > 0) {
                const totalSheets = data.available_sheets.length;
                const sheetList = data.available_sheets.slice(0, 10).join(", ");
                const more = data.available_sheets.length > 10 ? ` + ${data.available_sheets.length - 10} more` : "";
                infoHTML += `<br><span style="font-size: 0.85em; color: #6b7280;">📄 Template has ${totalSheets} sheets: ${sheetList}${more}</span>`;
            }

            sheetInfo.innerHTML = infoHTML;

            // Remove any existing sheet info first
            const existingInfo = document.getElementById("sheet-info-display");
            if (existingInfo) {
                existingInfo.remove();
            }

            // Insert at the top of result div
            resultDiv.insertBefore(sheetInfo, resultDiv.firstChild);
        }

        setDebug(`${modeName}: links ready`);
    }

    if (!values.length && !pdfUrl && !excelUrl) {
        setDebug(`${modeName}: done (no values/links in response)`);
    }

    // Cleanup busy state just in case
    setBusy(false);
}

// Robust click wiring:
// 1) normal handler
if (extractBtn) {
    extractBtn.addEventListener("click", (e) => {
        e.preventDefault();
        handleExtractOnly();
    });
}

// 2) document-level capture handler (works even if bubbling is weird)
document.addEventListener(
    "click",
    (e) => {
        const clickedExtract = e.target && e.target.closest && e.target.closest("#extractBtn");
        if (clickedExtract) {
            e.preventDefault();
            setDebug("Document-captured click on Extract Only");
            handleExtractOnly();
        }
    },
    true
);

setDebug("ready");

// Expose explicit handlers for inline onclick attributes
window.__extractOnlyClick = () => {
    setDebug("inline onclick: Extract Only");
    handleExtractOnly();
};

window.__extractPdfClick = () => {
    setDebug("inline onclick: Extract with PDF");
    handleExtractWithPdf();
};

window.__extractTemplateClick = () => {
    setDebug("inline onclick: Extract with Template");
    handleExtractWithTemplate();
};

// Prove that clicks/pointers are reaching JS at all
document.addEventListener(
    "pointerdown",
    (e) => {
        const id = e.target && e.target.id ? `#${e.target.id}` : e.target?.tagName;
        setDebug(`pointerdown on ${id || "(unknown)"}`);
    },
    true
);
