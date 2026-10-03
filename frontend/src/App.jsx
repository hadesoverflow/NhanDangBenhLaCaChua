import React, { useMemo, useState } from "react";
import { Activity, ImageUp, Leaf, Loader2, Search } from "lucide-react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = (import.meta.env.VITE_BACKEND_URL || "http://localhost:8010").replace(/\/$/, "");

function formatLabel(value) {
  if (!value) return "Unknown";
  if (value === "out_of_domain") return "Not a tomato leaf";
  if (value === "tomato_leaf") return "Tomato leaf";
  return value.replaceAll("Tomato__", "").replaceAll("Tomato_", "").replaceAll("_", " ");
}

function pct(value) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [settings, setSettings] = useState({
    gate_threshold: 0.6,
    detect_conf: 0.25,
    detect_iou: 0.45,
    max_detections: 3,
    include_probabilities: false,
  });

  const imageUrl = useMemo(() => preview, [preview]);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function chooseFile(event) {
    const nextFile = event.target.files?.[0];
    setResult(null);
    setError("");
    setFile(nextFile || null);
    setPreview(nextFile ? URL.createObjectURL(nextFile) : "");
  }

  async function predict() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", file);
    Object.entries(settings).forEach(([key, value]) => form.append(key, String(value)));

    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Request failed with status ${response.status}`);
      }
      setResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const gate = result?.gate || {};
  const probs = gate.probabilities || {};
  const classified = result?.final_status === "tomato_leaf_classified";

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <h1>Tomato Leaf Disease Pipeline</h1>
          <p>React frontend, FastAPI backend, ONNX Runtime inference.</p>
        </div>
        <div className="api-pill">
          <Activity size={16} />
          <span>{API_URL}</span>
        </div>
      </section>

      <section className="workspace">
        <aside className="controls">
          <label className="upload">
            <ImageUp size={22} />
            <span>{file ? file.name : "Choose an image"}</span>
            <input type="file" accept="image/*" onChange={chooseFile} />
          </label>

          <Control label="Tomato leaf threshold" value={settings.gate_threshold} min={0} max={1} step={0.05} onChange={(v) => updateSetting("gate_threshold", v)} />
          <Control label="YOLO confidence" value={settings.detect_conf} min={0} max={1} step={0.05} onChange={(v) => updateSetting("detect_conf", v)} />
          <Control label="YOLO IoU" value={settings.detect_iou} min={0} max={1} step={0.05} onChange={(v) => updateSetting("detect_iou", v)} />
          <Control label="Max detections" value={settings.max_detections} min={1} max={10} step={1} onChange={(v) => updateSetting("max_detections", v)} />

          <label className="check">
            <input type="checkbox" checked={settings.include_probabilities} onChange={(e) => updateSetting("include_probabilities", e.target.checked)} />
            <span>Show disease probabilities</span>
          </label>

          <button className="primary" disabled={!file || loading} onClick={predict}>
            {loading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
            <span>{loading ? "Running..." : "Predict"}</span>
          </button>
        </aside>

        <section className="viewer">
          {imageUrl ? (
            <div className="image-frame">
              <img
                src={imageUrl}
                alt="Uploaded preview"
                onLoad={(event) => {
                  setImageSize({
                    width: event.currentTarget.naturalWidth,
                    height: event.currentTarget.naturalHeight,
                  });
                }}
              />
              <BBoxOverlay detections={result?.detections || []} imageSize={imageSize} />
            </div>
          ) : (
            <div className="empty"><Leaf size={38} />Upload an image to begin</div>
          )}
        </section>

        <section className="result">
          <h2>Result</h2>
          {error && <div className="alert">{error}</div>}
          {!result && !error && <p className="muted">No prediction yet.</p>}
          {result && (
            <>
              <div className={classified ? "status ok" : "status warn"}>{result.final_status.replaceAll("_", " ")}</div>
              <div className="prediction">{classified ? formatLabel(result.best_disease_prediction) : result.message}</div>
              {classified && <Metric label="Disease confidence" value={pct(result.best_disease_confidence)} />}
              <div className="grid">
                <Metric label="Gate" value={formatLabel(gate.prediction)} />
                <Metric label="Tomato leaf" value={pct(probs.tomato_leaf)} />
                <Metric label="Out of domain" value={pct(probs.out_of_domain)} />
                <Metric label="Detections" value={String(result.detections?.length || 0)} />
              </div>
              <details>
                <summary>Raw JSON</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </>
          )}
        </section>
      </section>
    </main>
  );
}

function Control({ label, value, min, max, step, onChange }) {
  return (
    <label className="control">
      <span>{label}</span>
      <strong>{value}</strong>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function BBoxOverlay({ detections, imageSize }) {
  if (!detections.length || !imageSize.width || !imageSize.height) {
    return null;
  }

  return (
    <svg className="bbox-layer" viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="xMidYMid meet">
      {detections.map((detection, index) => {
        const [x1, y1, x2, y2] = detection.box_xyxy || [];
        if ([x1, y1, x2, y2].some((value) => typeof value !== "number")) {
          return null;
        }
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        const labelY = Math.max(18, y1 - 8);
        const label = `Leaf ${detection.rank || index + 1} ${pct(detection.detect_confidence)}`;
        return (
          <g key={`${detection.rank || index}-${x1}-${y1}`}>
            <rect className="bbox-rect" x={x1} y={y1} width={width} height={height} />
            <rect className="bbox-label-bg" x={x1} y={labelY - 18} width={Math.max(104, label.length * 8)} height="22" rx="4" />
            <text className="bbox-label" x={x1 + 6} y={labelY - 3}>{label}</text>
          </g>
        );
      })}
    </svg>
  );
}

createRoot(document.getElementById("root")).render(<App />);
