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
const status = $("status");

let uploadedImage = null;
let uploadedURL = null;
let activeModel = null; // {key, session, metadata}. Just one session in memory.
let loadPromise = null;
let working = false;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}
function setBusy(isBusy) {
  working = isBusy;
  classifyButton.disabled = isBusy || !uploadedImage;
  classifyButton.textContent = isBusy ? "Working…" : "✧  Classify landmark";
  modelSelect.disabled = isBusy;
}
function showProgress(percent = null) {
  $("progress-wrap").hidden = percent === null;
  $("progress-fill").style.width = `${Math.max(0, Math.min(100, percent ?? 0))}%`;
}
function clearResults() {
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
    const declaredSize = response.headers.get("Content-Length");
    const expectedPartSize = declaredSize === null ? null : Number(declaredSize);
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
    if (expectedPartSize !== null && partReceived !== expectedPartSize) {
      throw new Error(`Incomplete VGG16 part: ${part}`);
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
      renderResults(softmax(logits), metadata, urlAtClick, elapsed, modelKey);
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
