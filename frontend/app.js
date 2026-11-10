/* app.js — MediSense AI dashboard
   Unit 3: JavaScript fundamentals, DOM manipulation, Fetch API */

const API_BASE = window.MEDISENSE_API_BASE || "https://medisense-backend-025n.onrender.com/api";

const state = { patients: [] };

// ---------- View switching ----------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
    document.getElementById(`view-${btn.dataset.view}`).classList.remove("hidden");
    if (btn.dataset.view === "ranges") loadReferenceRanges();
  });
});

// ---------- API health ----------
async function checkHealth() {
  const dot = document.getElementById("apiDot");
  const label = document.getElementById("apiStatus");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("bad status");
    dot.className = "dot online";
    label.textContent = "API connected";
  } catch (err) {
    dot.className = "dot offline";
    label.textContent = "API unreachable";
  }
}

// ---------- Risk helpers ----------
function riskLevel(flags) {
  if (!flags || flags.length === 0) return "ok";
  const dangerFlags = ["HYPERTENSION", "LOW_SPO2", "TACHYCARDIA"];
  return flags.some((f) => dangerFlags.includes(f)) ? "danger" : "warn";
}

function vitalStatus(value, low, high) {
  if (value == null) return "ok";
  if (value < low || value > high) return "danger";
  const range = high - low;
  if (value < low + range * 0.1 || value > high - range * 0.1) return "warn";
  return "ok";
}

// ---------- Rendering ----------
function renderPatients() {
  const list = document.getElementById("patientList");
  list.innerHTML = "";

  if (state.patients.length === 0) {
    list.innerHTML = `<div class="empty-state">No patients yet — add one, or head to <strong>Import Data</strong> to bulk‑upload a workbook.</div>`;
  } else {
    state.patients.forEach((p) => list.appendChild(buildPatientRow(p)));
  }

  updateStats();
}

function buildPatientRow(patient) {
  const row = document.createElement("div");
  row.className = "patient-row";

  const flags = patient.records
    .flatMap((r) => computeFlags(r))
    .filter(Boolean);
  const level = riskLevel(flags);

  // --- name/meta cell ---
  const nameCell = document.createElement("div");
  nameCell.innerHTML = `
    <div class="patient-name">${escapeHtml(patient.name)}</div>
    <div class="patient-meta">${patient.patient_id} · ${patient.age}y · ${patient.gender}</div>
  `;

  // --- vitals strip cell ---
  const vitals = patient.records.find((r) => r.type === "VitalsRecord");
  const vitalsCell = document.createElement("div");
  if (vitals) {
    const ticks = [
      { label: "BP", val: `${vitals.systolic_bp}/${vitals.diastolic_bp}`, status: vitalStatus(vitals.systolic_bp, 90, 140) },
      { label: "HR", val: vitals.heart_rate, status: vitalStatus(vitals.heart_rate, 60, 100) },
      { label: "TMP", val: vitals.temperature_c, status: vitalStatus(vitals.temperature_c, 36.1, 37.9) },
      { label: "SpO2", val: vitals.spo2, status: vitalStatus(vitals.spo2, 94, 100) },
    ];
    const strip = document.createElement("div");
    strip.className = "vitals-strip";
    ticks.forEach((t) => {
      const el = document.createElement("div");
      el.className = `vtick ${t.status === "ok" ? "" : t.status}`;
      el.innerHTML = `<span class="val">${t.val}</span><span class="bar"></span><span>${t.label}</span>`;
      strip.appendChild(el);
    });
    vitalsCell.appendChild(strip);
  } else {
    vitalsCell.innerHTML = `<span class="no-vitals">No vitals recorded</span>`;
  }

  // --- risk cell ---
  const riskCell = document.createElement("div");
  const riskLabel = level === "ok" ? "Normal" : level === "warn" ? "Monitor" : "High risk";
  riskCell.innerHTML = `<span class="risk-pill ${level}">${riskLabel}</span>`;

  // --- delete cell ---
  const delCell = document.createElement("button");
  delCell.className = "row-delete";
  delCell.innerHTML = "✕";
  delCell.title = "Remove patient";
  delCell.addEventListener("click", () => deletePatient(patient.patient_id));

  row.append(nameCell, vitalsCell, riskCell, delCell);
  return row;
}

// Mirrors the polymorphic risk_flags() logic server-side, kept simple here
// since the API already returns a to_dict() without flags precomputed.
function computeFlags(record) {
  if (record.type !== "VitalsRecord") return [];
  const flags = [];
  if (record.systolic_bp >= 140 || record.diastolic_bp >= 90) flags.push("HYPERTENSION");
  if (record.heart_rate > 100) flags.push("TACHYCARDIA");
  if (record.heart_rate < 60) flags.push("BRADYCARDIA");
  if (record.temperature_c >= 38.0) flags.push("FEVER");
  if (record.spo2 < 94) flags.push("LOW_SPO2");
  return flags;
}

function updateStats() {
  const total = state.patients.length;
  let high = 0, normal = 0, ageSum = 0;
  state.patients.forEach((p) => {
    const flags = p.records.flatMap((r) => computeFlags(r));
    const level = riskLevel(flags);
    if (level === "danger") high += 1;
    if (flags.length === 0) normal += 1;
    ageSum += p.age;
  });
  document.getElementById("statTotal").textContent = total;
  document.getElementById("statHigh").textContent = high;
  document.getElementById("statNormal").textContent = normal;
  document.getElementById("statAvgAge").textContent = total ? Math.round(ageSum / total) : "–";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------- Fetch API calls ----------
async function loadPatients() {
  try {
    const res = await fetch(`${API_BASE}/patients`);
    state.patients = await res.json();
  } catch (err) {
    state.patients = [];
  }
  renderPatients();
}

async function deletePatient(id) {
  await fetch(`${API_BASE}/patients/${id}`, { method: "DELETE" });
  await loadPatients();
}

async function loadReferenceRanges() {
  const container = document.getElementById("rangeList");
  container.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const res = await fetch(`${API_BASE}/reference-ranges`);
    const ranges = await res.json();
    container.innerHTML = "";
    ranges.forEach((r) => {
      const row = document.createElement("div");
      row.className = "range-row";
      row.innerHTML = `<span>${r.test_name}</span><span>${r.low}</span><span>${r.high}</span><span>${r.unit}</span>`;
      container.appendChild(row);
    });
  } catch (err) {
    container.innerHTML = `<div class="empty-state">Could not load reference ranges — is the API running?</div>`;
  }
}

// ---------- Add patient modal ----------
const modal = document.getElementById("modalBackdrop");
document.getElementById("addPatientBtn").addEventListener("click", () => modal.classList.remove("hidden"));
document.getElementById("modalCancel").addEventListener("click", () => modal.classList.add("hidden"));

document.getElementById("addPatientForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const errorBox = document.getElementById("modalError");
  errorBox.classList.add("hidden");

  const data = Object.fromEntries(new FormData(form).entries());
  const payload = {
    patient_id: data.patient_id.trim(),
    name: data.name.trim(),
    age: Number(data.age),
    gender: data.gender,
    contact: data.contact.trim() || "N/A",
  };
  if (data.systolic_bp) {
    payload.vitals = {
      systolic_bp: Number(data.systolic_bp),
      diastolic_bp: Number(data.diastolic_bp),
      heart_rate: Number(data.heart_rate),
      temperature_c: Number(data.temperature_c),
      spo2: Number(data.spo2),
    };
  }

  try {
    const res = await fetch(`${API_BASE}/patients`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.message || "Failed to save patient");
    modal.classList.add("hidden");
    form.reset();
    await loadPatients();
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  }
});

// ---------- Import view: drag & drop + Fetch upload ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileChosen = document.getElementById("fileChosen");
const importSubmit = document.getElementById("importSubmit");
let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setSelectedFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setSelectedFile(fileInput.files[0]);
});

function setSelectedFile(file) {
  selectedFile = file;
  fileChosen.textContent = `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`;
  fileChosen.classList.remove("hidden");
  importSubmit.disabled = false;
}

document.getElementById("importForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  const resultBox = document.getElementById("importResult");
  resultBox.classList.remove("hidden");
  resultBox.innerHTML = "Importing…";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/patients/import`, { method: "POST", body: formData });
    const body = await res.json();
    if (!res.ok) throw new Error(body.message || "Import failed");

    resultBox.innerHTML = `
      <div class="summary">
        <span class="ok-count">${body.imported} imported</span>
        <span class="err-count">${body.failed} failed</span>
      </div>
      ${body.errors.length ? `<ul>${body.errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>` : ""}
    `;
    await loadPatients();
  } catch (err) {
    resultBox.innerHTML = `<div class="err-count">Error: ${escapeHtml(err.message)}</div>`;
  }
});

// ---------- Predict Risk view ----------
function riskBadgeClass(label) {
  if (label === "High") return "danger";
  if (label === "Medium") return "warn";
  return "ok";
}

function renderPredictionResult(label, source, inputs) {
  const card = document.getElementById("predictResultCard");
  const badgeClass = riskBadgeClass(label);
  const sourceText = source === "persisted"
    ? "Prediction from the persisted model (models/risk_classifier.joblib)"
    : "Prediction from an in-memory model (train via POST /api/ml/train)";
  const sourceClass = source === "persisted" ? "predict-source persisted" : "predict-source";

  card.innerHTML = `
    <div class="predict-result">
      <div class="predict-badge ${badgeClass}">
        <span class="dot-icon"></span>
        ${escapeHtml(label)} risk
      </div>
      <div class="${sourceClass}">${sourceText}</div>
      <div class="predict-inputs-echo">
        <div>Age<strong>${inputs.age}</strong></div>
        <div>BP<strong>${inputs.systolic_bp}/${inputs.diastolic_bp}</strong></div>
        <div>Heart rate<strong>${inputs.heart_rate}</strong></div>
        <div>Temp<strong>${inputs.temperature_c}°C</strong></div>
        <div>SpO2<strong>${inputs.spo2}%</strong></div>
      </div>
    </div>
  `;
}

document.getElementById("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorBox = document.getElementById("predictError");
  errorBox.classList.add("hidden");

  const data = Object.fromEntries(new FormData(e.target).entries());
  const payload = {
    age: Number(data.age),
    systolic_bp: Number(data.systolic_bp),
    diastolic_bp: Number(data.diastolic_bp),
    heart_rate: Number(data.heart_rate),
    temperature_c: Number(data.temperature_c),
    spo2: Number(data.spo2),
  };

  const submitBtn = document.getElementById("predictSubmit");
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";

  try {
    const res = await fetch(`${API_BASE}/ml/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.message || "Prediction failed");
    renderPredictionResult(body.predicted_risk, body.model_source, payload);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict risk";
  }
});

// ---------- Init ----------
checkHealth();
loadPatients();
setInterval(checkHealth, 15000);
