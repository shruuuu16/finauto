console.log("JavaScript loaded!");

const docInput = document.getElementById("document");
const tplInput = document.getElementById("template");
const docName = document.getElementById("docName");
const tplName = document.getElementById("tplName");

const extractBtn = document.getElementById("extractBtn");
const extractPdfBtn = document.getElementById("extractPdfBtn");
const extractTemplateBtn = document.getElementById("extractTemplateBtn");

const jobSection = document.getElementById("jobSection");
const jobIdEl = document.getElementById("jobId");
const statusEl = document.getElementById("status");
const resultDiv = document.getElementById("result");
const extractedValuesDiv = document.getElementById("extractedValues");
const valuesListEl = document.getElementById("valuesList");
const excelLink = document.getElementById("excelLink");
const pdfLink = document.getElementById("pdfLink");

console.log("All elements found:", {
  docInput, tplInput, extractBtn, extractPdfBtn, extractTemplateBtn,
  jobSection, jobIdEl, statusEl, resultDiv
});

let currentJobId = null;
let pollInterval = null;

// Show file names when selected
docInput.addEventListener("change", (e) => {
  docName.textContent = e.target.files[0]?.name || "";
});

tplInput.addEventListener("change", (e) => {
  tplName.textContent = e.target.files[0]?.name || "";
});

// Mode 1: Extract Only
extractBtn.addEventListener("click", async () => {
  console.log("Extract Only button clicked!");
  const docFile = docInput.files[0];
  if (!docFile) {
    alert("Please select a document first!");
    return;
  }

  console.log("Document selected:", docFile.name);
  const formData = new FormData();
  formData.append("document", docFile);

  try {
    console.log("Sending request to /extract-only...");
    const res = await fetch("http://127.0.0.1:8000/extract-only", {
      method: "POST",
      body: formData,
    });

    console.log("Response status:", res.status);
    if (!res.ok) {
      const error = await res.text();
      console.error("Upload failed:", error);
      alert("Upload failed: " + error);
      return;
    }

    const data = await res.json();
    console.log("Response data:", data);
    currentJobId = data.job_id;

    console.log("Setting job ID to:", currentJobId);
    jobIdEl.textContent = currentJobId;
    statusEl.textContent = "processing";

    console.log("Job section should now be visible!");
    console.log("Starting to poll status...");
    pollInterval = setInterval(checkStatus, 2000);
  } catch (error) {
    alert("Upload error: " + error.message);
    console.error("Upload error:", error);
  }
});

// Mode 2: Extract with PDF (no template)
extractPdfBtn.addEventListener("click", async () => {
  const docFile = docInput.files[0];
  if (!docFile) {
    alert("Please select a document first!");
    return;
  }

  const formData = new FormData();
  formData.append("document", docFile);
  // No template attached

  try {
    const res = await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const error = await res.text();
      alert("Upload failed: " + error);
      return;
    }

    const data = await res.json();
    currentJobId = data.job_id;

    jobIdEl.textContent = currentJobId;
    statusEl.textContent = "processing";

    pollInterval = setInterval(checkStatus, 2000);
  } catch (error) {
    alert("Upload error: " + error.message);
    console.error("Upload error:", error);
  }
});

// Mode 3: Extract with Template
extractTemplateBtn.addEventListener("click", async () => {
  const docFile = docInput.files[0];
  const tplFile = tplInput.files[0];

  if (!docFile) {
    alert("Please select a document first!");
    return;
  }

  if (!tplFile) {
    alert("Please select an Excel template for this mode!");
    return;
  }

  const formData = new FormData();
  formData.append("document", docFile);
  formData.append("template", tplFile);

  try {
    const res = await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const error = await res.text();
      alert("Upload failed: " + error);
      return;
    }

    const data = await res.json();
    currentJobId = data.job_id;

    jobIdEl.textContent = currentJobId;
    statusEl.textContent = "processing";

    pollInterval = setInterval(checkStatus, 2000);
  } catch (error) {
    alert("Upload error: " + error.message);
    console.error("Upload error:", error);
  }
});

async function checkStatus() {
  console.log("Checking status for job:", currentJobId);
  const res = await fetch(`http://127.0.0.1:8000/status/${currentJobId}`);
  const data = await res.json();
  console.log("Status response:", data);

  statusEl.textContent = data.status;

  if (data.status === "done") {
    clearInterval(pollInterval);
    console.log("Job completed!");

    // Show message if available
    if (data.message) {
      statusEl.textContent = data.status + " - " + data.message;
    }
    // Show extracted values if available (extract-only mode)
    if (data.ocr_values && data.ocr_values.length > 0) {
      console.log("Showing extracted values:", data.ocr_values.length);
      extractedValuesDiv.style.display = "block";
      valuesListEl.innerHTML = "";
      data.ocr_values.forEach((value, idx) => {
        const p = document.createElement("p");
        p.className = "extracted-value";
        p.textContent = `${idx + 1}. ${value}`;
        valuesListEl.appendChild(p);
      });
    }

    // Show download links if available
    if (data.excel || data.pdf) {
      console.log("Showing download links");
      resultDiv.style.display = "block";

      if (data.excel) {
        excelLink.href = `http://127.0.0.1:8000/${data.excel.replace(/\\/g, '/')}`;
        excelLink.style.display = 'inline-block';
      }

      if (data.pdf) {
        pdfLink.href = `http://127.0.0.1:8000/${data.pdf.replace(/\\/g, '/')}`;
        pdfLink.style.display = 'inline-block';
      }
    }
  }

  if (data.status === "failed") {
    clearInterval(pollInterval);
    console.error("Job failed:", data.error);
    alert("Processing failed: " + (data.error || "Unknown error"));
  }
}
