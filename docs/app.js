// JordanHeritageVision — four-model browser inference with ONNX Runtime Web.
// Only the selected model is downloaded; the previous ONNX session is released
// before loading another model to limit browser memory use.
// Uploaded photos are processed locally and never submitted to a prediction server.

const ORT_VERSION = "1.31.1"; // Match the pinned script in index.html.
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const MODEL_CONFIG = Object.freeze({
  efficientnet_b0: {
    name: "EfficientNetB0",
    onnx: new URL("./model/efficientnet_b0.onnx", import.meta.url),
    metadata: new URL("./model/model_metadata.json", import.meta.url),
    testAccuracy: "91.57%",
    note: "EfficientNetB0: smaller, faster browser model (15.9 MiB ONNX).",
  },
  resnet101: {
    name: "ResNet101",
    onnx: new URL("./model/resnet101.onnx", import.meta.url),
    metadata: new URL("./model/resnet101_metadata.json", import.meta.url),
    testAccuracy: "92.46%",
    note: "ResNet101: best held-out test accuracy; larger download (81.5 MiB ONNX). May be slower on mobile.",
  },
  inception_v3: {
    name: "InceptionV3",
    onnx: new URL("./model/inception_v3.onnx", import.meta.url),
    metadata: new URL("./model/inception_v3_metadata.json", import.meta.url),
    testAccuracy: "88.91%",
    note: "InceptionV3: 299 x 299 input, 42.0 MiB ONNX. This model may be slower on mobile.",
  },
  vgg16: {
    name: "VGG16",
    metadata: new URL("./model/vgg16_metadata.json", import.meta.url),
    testAccuracy: "87.36%",
    note: "VGG16: 224 x 224 input. Large 256.15 MiB model downloaded in 7 parts. Desktop recommended; may not run on mobile.",
  },
});

const $ = (id) => document.getElementById(id);
const input = $("photo-input");
const dropzone = $("dropzone");
const preview = $("upload-preview");
const modelSelect = $("model-select");
const classifyButton = $("classify-btn");
const compareButton = $("compare-btn");
const status = $("status");

let uploadedImage = null;
let uploadedURL = null;
let activeModel = null; // {key, session, metadata}. Just one session in memory.
let loadPromise = null;
let working = false;
let lastPrediction = null; // The selected model result for this uploaded image.
const comparisonResults = new Map();

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}
function setBusy(isBusy) {
  working = isBusy;
  classifyButton.disabled = isBusy || !uploadedImage;
  classifyButton.textContent = isBusy ? "Working…" : "✧  Classify landmark";
  modelSelect.disabled = isBusy;
  compareButton.disabled = isBusy || !lastPrediction;
}
function showProgress(percent = null) {
  $("progress-wrap").hidden = percent === null;
  $("progress-fill").style.width = `${Math.max(0, Math.min(100, percent ?? 0))}%`;
}
function clearResults() {
  lastPrediction = null;
  compareButton.disabled = true;
  clearComparison();
  $("result-content").hidden = true;
  $("result-empty").hidden = false;
}
function updateModelDescription() {
  const conf = MODEL_CONFIG[modelSelect.value];
  if (!conf) return;
  $("model-note").textContent = conf.note;
  $("selected-accuracy").textContent = conf.testAccuracy;
}
modelSelect.addEventListener("change", () => {
  clearResults();
  updateModelDescription();
  showProgress(null);
  setStatus(uploadedImage ? "Model selected. Click Classify landmark." : "Upload an image to begin.");
});
updateModelDescription();

async function selectPhoto(file) {
  if (working || !file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Please choose an image file (JPG, PNG, or WEBP).", true);
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    setStatus("Image is larger than 15 MB. Choose a smaller file.", true);
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  try {
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("The image could not be decoded."));
      image.src = url;
    });
    if (!image.naturalWidth || !image.naturalHeight) throw new Error("Invalid image dimensions.");
  } catch (error) {
    URL.revokeObjectURL(url);
    setStatus(error.message, true);
    return;
  }
  if (uploadedURL) URL.revokeObjectURL(uploadedURL);
  uploadedURL = url;
  uploadedImage = image;
  preview.src = url;
  preview.hidden = false;
  $("drop-inner").hidden = true;
  dropzone.classList.add("has-photo");
  $("file-label").textContent = `${file.name} · ${image.naturalWidth} × ${image.naturalHeight}`;
  clearResults();
  setBusy(false);
  showProgress(null);
  setStatus("Image ready. Click Classify landmark.");
}

dropzone.addEventListener("click", () => { if (!working) input.click(); });
dropzone.addEventListener("keydown", (event) => {
  if (!working && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    input.click();
  }
});
input.addEventListener("change", () => {
  void selectPhoto(input.files?.[0]);
  input.value = "";
});
["dragenter", "dragover", "dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    if (working) return;
    if (name === "dragenter" || name === "dragover") dropzone.classList.add("dragover");
    else dropzone.classList.remove("dragover");
    if (name === "drop") void selectPhoto(event.dataTransfer?.files?.[0]);
  });
});

function validateMetadata(meta, modelKey) {
  const expectedClasses = ["Ajloun", "Jerash", "Petra", "RomanAmphitheater", "UmmQais", "WadiRum"];
  // Each model has its own evaluation resolution; do not force 224x224.
  const expectedSize = modelKey === "inception_v3" ? 299 : 224;
  const expectedResize = modelKey === "inception_v3" ? 342 : 256;
  const expectedMean = [0.485, 0.456, 0.406];
  const expectedStd = [0.229, 0.224, 0.225];
  const matchesValues = (actual, expected) =>
    Array.isArray(actual) && actual.length === expected.length &&
    actual.every((n, i) => Number.isFinite(n) && Math.abs(n - expected[i]) < 1e-6);

  if (meta.model_name !== modelKey ||
      meta.image_size !== expectedSize || meta.resize_short_side !== expectedResize ||
      meta.crop_type !== "center" ||
      !Array.isArray(meta.classes) || meta.classes.join("|") !== expectedClasses.join("|") ||
      !matchesValues(meta.mean, expectedMean) || !matchesValues(meta.std, expectedStd) ||
      (modelKey === "inception_v3" && meta.transform_input !== true) ||
      meta.input_layout !== "NCHW" || meta.input_dtype !== "float32" ||
      meta.onnx_input !== "images" || meta.onnx_output !== "logits") {
    throw new Error(`Invalid preprocessing metadata for ${modelKey}.`);
  }
  // InceptionV3's transform_input is inside the exported model graph.
  // Do NOT apply it again in JavaScript.
  if (modelKey === "vgg16") {
    const expectedPartNames = Array.from({ length: 7 }, (_, index) =>
      `vgg16.onnx.part${String(index).padStart(3, "0")}`
    );
    if (
      !Array.isArray(meta.onnx_parts) ||
      meta.onnx_parts.join("|") !== expectedPartNames.join("|") ||
      !Number.isSafeInteger(meta.onnx_bytes) ||
      meta.onnx_bytes !== 268588199 ||
      meta.onnx_sha256 !== "8fc420a96d7550c587d4334b67110900ca9f0bb8cfc1b86fe916a38903b4d324"
    ) {
      throw new Error("Invalid VGG16 model-parts metadata.");
    }
  }
}

async function fetchModelWithProgress(modelUrl, modelName) {
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`${modelName} download failed: HTTP ${response.status}. Check docs/model/.`);
  const total = Number(response.headers.get("Content-Length"));
  const reader = response.body?.getReader();
  if (!reader) return new Uint8Array(await response.arrayBuffer());

  const chunks = [];
  let count = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    count += value.byteLength;
    if (total) showProgress(2 + count / total * 85);
    setStatus(`Downloading ${modelName}… ${(count / 1048576).toFixed(1)} MiB received`);
  }
  const bytes = new Uint8Array(count);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

// Stream the seven VGG16 parts into one preallocated buffer.
// Avoid keeping seven separate ArrayBuffers in memory.
async function fetchMultipartModel(conf, meta) {
  const bytes = new Uint8Array(meta.onnx_bytes);
  let received = 0;
  let lastDisplay = 0;

  for (let i = 0; i < meta.onnx_parts.length; i++) {
    const part = meta.onnx_parts[i];
    const partUrl = new URL(part, conf.metadata);
    const response = await fetch(partUrl);
    if (!response.ok) {
      throw new Error(`VGG16 part ${i + 1}/${meta.onnx_parts.length} failed: HTTP ${response.status}. Check docs/model/${part}`);
    }
    // Fetch transparently decodes HTTP compression, so Content-Length can
    // describe transferred bytes rather than the uncompressed ONNX bytes.
    // Validate against the actual chunk sizes produced by the export script.
    const chunkSize = 40 * 1024 * 1024;
    const expectedPartSize = i === meta.onnx_parts.length - 1
      ? meta.onnx_bytes - chunkSize * (meta.onnx_parts.length - 1)
      : chunkSize;
    let partReceived = 0;
    const appendChunk = (chunk) => {
      if (received + chunk.byteLength > bytes.byteLength) {
        throw new Error(`VGG16 part ${i + 1} exceeds expected model size.`);
      }
      bytes.set(chunk, received);
      received += chunk.byteLength;
      partReceived += chunk.byteLength;
      if (received - lastDisplay >= 2 * 1024 * 1024 || received === meta.onnx_bytes) {
        showProgress(2 + received / meta.onnx_bytes * 88);
        setStatus(`Downloading VGG16 part ${i + 1}/${meta.onnx_parts.length}… ${(received / 1048576).toFixed(1)} / ${(meta.onnx_bytes / 1048576).toFixed(1)} MiB`);
        lastDisplay = received;
      }
    };

    if (response.body?.getReader) {
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        appendChunk(value);
      }
    } else {
      appendChunk(new Uint8Array(await response.arrayBuffer()));
    }
    if (partReceived !== expectedPartSize) {
      throw new Error(`Incomplete VGG16 part: ${part}. Received ${partReceived} bytes; expected ${expectedPartSize}. Check the deployed file and browser cache.`);
    }
  }
  if (received !== meta.onnx_bytes) {
    throw new Error(`VGG16 model is incomplete: received ${received} of ${meta.onnx_bytes} bytes.`);
  }
  return bytes;
}

async function loadSelectedModel(modelKey) {
  if (activeModel?.key === modelKey) return activeModel;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    const conf = MODEL_CONFIG[modelKey];
    if (!conf) throw new Error("Unknown model selection.");
    if (!window.ort?.InferenceSession) {
      throw new Error("ONNX Runtime Web is not available. Check your internet connection.");
    }

    // Keep a single CNN in browser memory at once.
    if (activeModel) {
      setStatus("Releasing previous model from memory…");
      await activeModel.session.release();
      activeModel = null;
    }

    setStatus(`Reading ${conf.name} metadata…`);
    const response = await fetch(conf.metadata);
    if (!response.ok) throw new Error(`Metadata download failed: HTTP ${response.status}`);
    const meta = await response.json();
    validateMetadata(meta, modelKey);

    window.ort.env.wasm.wasmPaths = ORT_DIST;
    window.ort.env.wasm.numThreads = 1; // Safe without cross-origin isolation.
    window.ort.env.wasm.proxy = false;
    showProgress(2);
    const bytes = modelKey === "vgg16"
      ? await fetchMultipartModel(conf, meta)
      : await fetchModelWithProgress(conf.onnx, conf.name);
    showProgress(94);
    setStatus(`Initializing ${conf.name} in your browser…`);
    const loadedSession = await window.ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    if (!loadedSession.inputNames.includes(meta.onnx_input) ||
        !loadedSession.outputNames.includes(meta.onnx_output)) {
      await loadedSession.release();
      throw new Error("ONNX input/output names do not match metadata.");
    }
    activeModel = { key: modelKey, session: loadedSession, metadata: meta };
    showProgress(100);
    return activeModel;
  })();

  try { return await loadPromise; }
  finally { loadPromise = null; }
}

function preprocessImage(image, meta) {
  const shorter = Math.min(image.naturalWidth, image.naturalHeight);
  const factor = meta.resize_short_side / shorter;
  const scaledW = Math.max(meta.image_size, Math.round(image.naturalWidth * factor));
  const scaledH = Math.max(meta.image_size, Math.round(image.naturalHeight * factor));
  const resized = document.createElement("canvas");
  resized.width = scaledW;
  resized.height = scaledH;
  const resizedCtx = resized.getContext("2d", { willReadFrequently: true });
  resizedCtx.imageSmoothingEnabled = true;
  resizedCtx.imageSmoothingQuality = "high";
  resizedCtx.drawImage(image, 0, 0, scaledW, scaledH);
  const size = meta.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const left = Math.round((scaledW - size) / 2);
  const top = Math.round((scaledH - size) / 2);
  ctx.drawImage(resized, left, top, size, size, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const pixels = size * size;
  const nchw = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    const p = i * 4;
    for (let channel = 0; channel < 3; channel++) {
      nchw[channel * pixels + i] = (rgba[p + channel] / 255 - meta.mean[channel]) / meta.std[channel];
    }
  }
  return new window.ort.Tensor("float32", nchw, [1, 3, size, size]);
}

function softmax(logits) {
  const values = Array.from(logits);
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

function renderResults(scores, meta, imageURL, elapsed, modelKey) {
  const pairs = meta.classes.map((name, index) => ({ name, probability: scores[index] }));
  pairs.sort((a, b) => b.probability - a.probability);
  const best = pairs[0];
  $("result-empty").hidden = true;
  $("result-content").hidden = false;
  $("prediction-name").textContent = best.name;
  $("prediction-confidence").textContent = `${(best.probability * 100).toFixed(1)}% score`;
  $("result-photo").src = imageURL;
  const scoresContainer = $("scores");
  scoresContainer.replaceChildren();
  for (let i = 0; i < pairs.length; i++) {
    const item = pairs[i];
    const row = document.createElement("div");
    row.className = "score-item" + (i === 0 ? " winner" : "");
    const head = document.createElement("div");
    head.className = "score-head";
    const name = document.createElement("span");
    name.textContent = item.name;
    const percent = document.createElement("span");
    percent.textContent = `${(item.probability * 100).toFixed(2)}%`;
    head.append(name, percent);
    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = `${Math.max(0, Math.min(100, item.probability * 100))}%`;
    track.append(fill);
    row.append(head, track);
    scoresContainer.append(row);
  }
  $("inference-detail").textContent =
    `Model: ${MODEL_CONFIG[modelKey].name} · Browser CPU/WASM · Inference: ${elapsed.toFixed(0)} ms · Softmax scores are not calibrated probabilities.`;
}

classifyButton.addEventListener("click", async () => {
  if (!uploadedImage || working) return;
  clearComparison();
  lastPrediction = null;
  const imageAtClick = uploadedImage;
  const urlAtClick = uploadedURL;
  const modelKey = modelSelect.value;
  setBusy(true);
  try {
    const { session, metadata } = await loadSelectedModel(modelKey);
    setStatus(`Running ${MODEL_CONFIG[modelKey].name} inference…`);
    const tensor = preprocessImage(imageAtClick, metadata);
    const started = performance.now();
    const output = await session.run({ [metadata.onnx_input]: tensor });
    const elapsed = performance.now() - started;
    const logits = output[metadata.onnx_output]?.data;
    if (!logits || logits.length !== metadata.classes.length) {
      throw new Error("The model returned an unexpected output shape.");
    }
    if (uploadedImage === imageAtClick && modelSelect.value === modelKey) {
      const scores = softmax(logits);
      renderResults(scores, metadata, urlAtClick, elapsed, modelKey);
      lastPrediction = { modelKey, image: imageAtClick, scores, elapsed, classes: [...metadata.classes] };
      $("compare-hint").textContent = `Ready to compare. Your ${MODEL_CONFIG[modelKey].name} result will be reused.`;
      setStatus("Classification completed on your device.");
    }
  } catch (error) {
    console.error(error);
    setStatus(`Classification failed: ${error.message}`, true);
  } finally {
    showProgress(null);
    setBusy(false);
  }
});


// -----------------------------------------
// MODEL COMPARISON — v2.0
// Keep the original one-model prediction visible above the dashboard.
// Load only one additional ONNX model at a time to limit memory use.
// -----------------------------------------
const MODEL_ORDER = ["efficientnet_b0", "resnet101", "inception_v3", "vgg16"];

function clearComparison() {
  comparisonResults.clear();
  $("comparison-dashboard").hidden = true;
  $("comparison-grid").replaceChildren();
  $("comparison-speed").replaceChildren();
  $("comparison-summary").textContent = "";
  $("comparison-status").textContent = "";
  $("compare-hint").textContent = "Run a single-model prediction above to unlock the comparison.";
}

function makeText(tag, className, value) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  el.textContent = value;
  return el;
}

function makeComparisonCard(modelKey, selectedKey) {
  const conf = MODEL_CONFIG[modelKey];
  const card = document.createElement("article");
  card.className = "compare-card" + (modelKey === selectedKey ? " selected" : "");
  card.dataset.model = modelKey;

  const heading = makeText("h3", "", conf.name);
  const top = document.createElement("div");
  top.className = "compare-card-head";
  top.append(heading);
  if (modelKey === selectedKey) {
    top.append(makeText("span", "compare-selected", "Your selection"));
  }
  const body = makeText("div", "compare-card-body", "Waiting to run…");
  card.append(top, body);
  return card;
}

function showComparisonResult(modelKey, result) {
  comparisonResults.set(modelKey, result);
  const card = Array.from($("comparison-grid").children)
    .find((node) => node.dataset.model === modelKey);
  if (!card) return;
  const body = card.querySelector(".compare-card-body");
  body.replaceChildren();

  if (result.error) {
    body.append(makeText("p", "compare-error", `Could not run: ${result.error}`));
    return;
  }
  const ranked = result.classes.map((name, i) => ({name, score: result.scores[i]}))
    .sort((a, b) => b.score - a.score);
  const best = ranked[0];
  const prediction = document.createElement("div");
  prediction.className = "compare-prediction";
  prediction.append(
    makeText("strong", "compare-predicted-name", best.name),
    makeText("span", "compare-score", `${(best.score * 100).toFixed(1)}% score`)
  );
  body.append(prediction);

  const metrics = document.createElement("div");
  metrics.className = "compare-metrics";
  metrics.append(
    makeText("div", "", `Inference: ${result.elapsed.toFixed(0)} ms`),
    makeText("div", "", `Test accuracy: ${MODEL_CONFIG[modelKey].testAccuracy}`)
  );
  body.append(metrics);

  const details = document.createElement("details");
  details.className = "compare-details";
  details.append(makeText("summary", "", "Show all 6 class scores"));
  for (const item of ranked) {
    const line = document.createElement("div");
    line.className = "compare-class-line";
    const head = document.createElement("div");
    head.className = "compare-class-head";
    head.append(makeText("span", "", item.name), makeText("span", "", `${(item.score * 100).toFixed(2)}%`));
    const track = document.createElement("div");
    track.className = "compare-class-track";
    const fill = document.createElement("div");
    fill.className = "compare-class-fill";
    fill.style.width = `${Math.min(100, Math.max(0, item.score * 100))}%`;
    track.append(fill);
    line.append(head, track);
    details.append(line);
  }
  body.append(details);
}

function updateComparisonSummary() {
  const success = [...comparisonResults.values()].filter((r) => !r.error);
  const failed = comparisonResults.size - success.length;
  if (!success.length) {
    $("comparison-summary").textContent = "No models have returned a prediction yet.";
    return;
  }
  const counts = new Map();
  for (const r of success) {
    const winnerIndex = r.scores.indexOf(Math.max(...r.scores));
    const label = r.classes[winnerIndex];
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  const agreement = [...counts.entries()].sort((a,b)=>b[1]-a[1])[0];
  const complete = comparisonResults.size === MODEL_ORDER.length;
  let label;
  if (success.length === 4 && agreement[1] === 4) {
    label = `All 4 models agree: ${agreement[0]}.`;
  } else if (complete) {
    label = `Most common prediction: ${agreement[0]} (${agreement[1]} of ${success.length} successful models).`;
  } else {
    label = `${success.length} of 4 models completed. Current leading label: ${agreement[0]}.`;
  }
  if (complete && failed) label += ` ${failed} model(s) failed; see cards.`;
  $("comparison-summary").textContent = label;
}

function updateSpeedChart() {
  const region = $("comparison-speed");
  region.replaceChildren();
  const successful = MODEL_ORDER.map((key) => ({key, value: comparisonResults.get(key)}))
    .filter((x) => x.value && !x.value.error);
  if (!successful.length) return;
  const maxMs = Math.max(1, ...successful.map((x) => x.value.elapsed));
  region.append(makeText("h3", "compare-speed-title", "Browser inference time"));
  for (const {key, value} of successful) {
    const row = document.createElement("div");
    row.className = "compare-speed-row";
    const label = makeText("span", "compare-speed-name", MODEL_CONFIG[key].name);
    const bar = document.createElement("div");
    bar.className = "compare-speed-track";
    const fill = document.createElement("div");
    fill.className = "compare-speed-fill";
    fill.style.width = `${value.elapsed / maxMs * 100}%`;
    bar.append(fill);
    row.append(label, bar, makeText("span", "compare-speed-ms", `${value.elapsed.toFixed(0)} ms`));
    region.append(row);
  }
  region.append(makeText("p", "compare-footnote", "Inference only; excludes model downloads and initialization. Speeds vary by device and run."));
}

compareButton.addEventListener("click", async () => {
  if (working || !uploadedImage || !lastPrediction) return;
  if (lastPrediction.image !== uploadedImage || lastPrediction.modelKey !== modelSelect.value) return;

  // Snapshot the single prediction before starting asynchronous loading.
  const original = {...lastPrediction};
  const sameImage = original.image;
  setBusy(true);
  comparisonResults.clear();
  $("comparison-grid").replaceChildren();
  $("comparison-speed").replaceChildren();
  $("comparison-dashboard").hidden = false;
  $("comparison-summary").textContent = "Starting model comparison…";
  $("comparison-status").textContent = "Preparing models…";
  for (const key of MODEL_ORDER) {
    $("comparison-grid").append(makeComparisonCard(key, original.modelKey));
  }
  // The selected result is already on screen. Do not rerun it.
  showComparisonResult(original.modelKey, original);
  updateComparisonSummary();
  updateSpeedChart();

  const remaining = MODEL_ORDER.filter((key) => key !== original.modelKey);
  for (let i = 0; i < remaining.length; i++) {
    const key = remaining[i];
    const conf = MODEL_CONFIG[key];
    $("comparison-status").textContent = `Running ${conf.name} (${i+1} of ${remaining.length} additional models). Loading weights may take time…`;
    const card = Array.from($("comparison-grid").children)
      .find((item) => item.dataset.model === key);
    card.querySelector(".compare-card-body").textContent = "Downloading and running…";
    try {
      const {session, metadata} = await loadSelectedModel(key);
      if (sameImage !== uploadedImage) throw new Error("Image changed during comparison.");
      const tensor = preprocessImage(sameImage, metadata);
      const started = performance.now();
      const output = await session.run({[metadata.onnx_input]: tensor});
      const elapsed = performance.now() - started;
      const logits = output[metadata.onnx_output]?.data;
      if (!logits || logits.length !== metadata.classes.length) {
        throw new Error("Unexpected model output shape.");
      }
      showComparisonResult(key, {scores: softmax(logits), classes: [...metadata.classes], elapsed});
    } catch (error) {
      console.error(`Compare ${conf.name}:`, error);
      showComparisonResult(key, {error: error.message || String(error)});
    }
    updateComparisonSummary();
    updateSpeedChart();
  }
  const failureCount = [...comparisonResults.values()].filter((x) => x.error).length;
  $("comparison-status").textContent = failureCount
    ? `Comparison finished with ${failureCount} error(s). Your original prediction is unchanged.`
    : "Comparison completed. Your original prediction remains visible above.";
  setStatus("Classification completed on your device.");
  showProgress(null);
  setBusy(false);
});
