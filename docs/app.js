// JordanHeritageVision browser inference. No backend, no image uploads.
// ONNX Runtime Web and WASM assets are pinned to the same release.
const ORT_VERSION = "1.31.1";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_URL = new URL("./model/efficientnet_b0.onnx", import.meta.url);
const METADATA_URL = new URL("./model/model_metadata.json", import.meta.url);

const $ = (id) => document.getElementById(id);
const input = $("photo-input");
const dropzone = $("dropzone");
const preview = $("upload-preview");
const classifyButton = $("classify-btn");
const status = $("status");
let uploadedImage = null;
let uploadedURL = null;
let session = null;
let metadata = null;
let loadPromise = null;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}
function setBusy(isBusy) {
  classifyButton.disabled = isBusy || !uploadedImage;
  classifyButton.textContent = isBusy ? "Working…" : "✧  Classify landmark";
}
function showProgress(percent = null) {
  $("progress-wrap").hidden = percent === null;
  $("progress-fill").style.width = `${Math.max(0, Math.min(100, percent ?? 0))}%`;
}
function clearResults() {
  $("result-content").hidden = true;
  $("result-empty").hidden = false;
}

async function selectPhoto(file) {
  if (!file) return;
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

dropzone.addEventListener("click", () => input.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
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
    if (name === "dragenter" || name === "dragover") dropzone.classList.add("dragover");
    else dropzone.classList.remove("dragover");
    if (name === "drop") void selectPhoto(event.dataTransfer?.files?.[0]);
  });
});

async function loadModelOnce() {
  if (session && metadata) return { session, metadata };
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    if (!window.ort?.InferenceSession) {
      throw new Error("Could not load ONNX Runtime Web. Check your internet connection and reload.");
    }
    setStatus("Reading model metadata…");
    const metaResponse = await fetch(METADATA_URL);
    if (!metaResponse.ok) throw new Error(`Metadata download failed: HTTP ${metaResponse.status}`);
    const meta = await metaResponse.json();
    validateMetadata(meta);

    window.ort.env.wasm.wasmPaths = ORT_DIST;
    window.ort.env.wasm.numThreads = 1; // GitHub Pages need not be cross-origin isolated.
    window.ort.env.wasm.proxy = false;
    setStatus("Downloading model to your browser (first run may take longer)…");
    showProgress(2);
    const modelResponse = await fetch(MODEL_URL);
    if (!modelResponse.ok) throw new Error(`Model download failed: HTTP ${modelResponse.status}`);
    const contentLength = Number(modelResponse.headers.get("Content-Length"));
    const reader = modelResponse.body?.getReader();
    let bytes;
    if (reader) {
      const chunks = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        if (contentLength) showProgress(2 + (received / contentLength) * 85);
        setStatus(`Downloading model… ${(received / 1048576).toFixed(1)} MB received`);
      }
      bytes = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    } else {
      bytes = new Uint8Array(await modelResponse.arrayBuffer());
    }
    setStatus("Initializing ONNX inference engine…");
    showProgress(94);
    const loadedSession = await window.ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    if (!loadedSession.inputNames.includes(meta.onnx_input)) throw new Error("ONNX input name does not match metadata.");
    if (!loadedSession.outputNames.includes(meta.onnx_output)) throw new Error("ONNX output name does not match metadata.");
    session = loadedSession;
    metadata = meta;
    showProgress(100);
    return { session, metadata };
  })();
  try { return await loadPromise; }
  finally { loadPromise = null; }
}

function validateMetadata(meta) {
  if (meta.image_size !== 224 || meta.resize_short_side !== 256 ||
      !Array.isArray(meta.classes) || meta.classes.length !== 6 ||
      !Array.isArray(meta.mean) || meta.mean.length !== 3 ||
      !Array.isArray(meta.std) || meta.std.length !== 3 ||
      meta.input_layout !== "NCHW" || meta.input_dtype !== "float32" ||
      typeof meta.onnx_input !== "string" || typeof meta.onnx_output !== "string") {
    throw new Error("Unsupported or incomplete preprocessing metadata.");
  }
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

function renderResults(scores, meta, imageURL, elapsed) {
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
  $("inference-detail").textContent = `Model: EfficientNetB0 · Browser CPU/WASM · Inference: ${elapsed.toFixed(0)} ms · Softmax scores are not calibrated probabilities.`;
}

classifyButton.addEventListener("click", async () => {
  if (!uploadedImage) return;
  const imageAtClick = uploadedImage;
  const urlAtClick = uploadedURL;
  setBusy(true);
  try {
    const { session: activeSession, metadata: meta } = await loadModelOnce();
    setStatus("Preparing image and running inference…");
    const tensor = preprocessImage(imageAtClick, meta);
    const started = performance.now();
    const feeds = { [meta.onnx_input]: tensor };
    const output = await activeSession.run(feeds);
    const elapsed = performance.now() - started;
    const logits = output[meta.onnx_output]?.data;
    if (!logits || logits.length !== meta.classes.length) throw new Error("The model returned an unexpected output shape.");
    const probabilities = softmax(logits);
    // Do not replace results if the user selected a different image mid-inference.
    if (uploadedImage === imageAtClick) {
      renderResults(probabilities, meta, urlAtClick, elapsed);
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
