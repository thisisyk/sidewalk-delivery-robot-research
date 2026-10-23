/* Run the unchanged Python simulator and exported frozen HistGB in a worker.
 * No saved trajectory is used as a substitute for a requested computation.
 * The UI can cancel a synchronous Python run by terminating this worker.
 */
const PYODIDE_VERSION = "314.0.7";
const BASE_URL = new URL(".", self.location.href);
const MODULE_NAMES = [
  "__init__", "browser_model", "geometry", "samples", "simulation",
  "urban_worlds", "urban_simulation", "sidewalk_worlds", "assistance",
  "sidewalk_simulation",
].map(name => `delivery_risk/${name}.py`);
let initialization = null;
let pyodide = null;
let manifest = null;
let manifestSha256 = null;
let info = null;
let busy = false;

function progress(requestId, phase, message, extra = {}) {
  self.postMessage({ type: "progress", requestId, phase, message, ...extra });
}

function localURL(path) {
  if (typeof path !== "string" || !path) throw new Error("invalid_resource_url");
  const url = new URL(path, BASE_URL);
  if (url.origin !== BASE_URL.origin || !url.pathname.startsWith(BASE_URL.pathname)) {
    throw new Error("runtime_resources_must_be_same_origin_and_site_directory");
  }
  return url;
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, "0")).join("");
}

async function fetchBytes(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(url, { cache: "no-cache", signal: controller.signal });
    if (!response.ok) throw new Error(`resource_download_failed:${response.status}:${url.pathname}`);
    return new Uint8Array(await response.arrayBuffer());
  } finally {
    clearTimeout(timer);
  }
}

async function verifiedResource(record) {
  if (!record || !/^[a-f0-9]{64}$/.test(record.sha256 || "")) {
    throw new Error("missing_resource_sha256");
  }
  const bytes = await fetchBytes(localURL(record.url));
  if (await sha256(bytes) !== record.sha256) throw new Error(`resource_sha256_mismatch:${record.url}`);
  return bytes;
}

function validateManifest(candidate) {
  if (!candidate || candidate.runtime?.version !== PYODIDE_VERSION) {
    throw new Error("unsupported_python_runtime_version");
  }
  const names = (candidate.modules || []).map(record => record.name).sort();
  if (JSON.stringify(names) !== JSON.stringify([...MODULE_NAMES].sort())) {
    throw new Error("exact_python_module_manifest_required");
  }
  localURL(candidate.runtime.indexURL);
  if (!candidate.model || !candidate.bootstrap) throw new Error("incomplete_compute_manifest");
}

async function initialize(requestId) {
  progress(requestId, "manifest", "正在读取模型和计算模块清单…");
  const manifestBytes = await fetchBytes(localURL("compute-manifest.json"));
  manifestSha256 = await sha256(manifestBytes);
  manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  validateManifest(manifest);
  const runtimeURL = localURL(manifest.runtime.indexURL);
  if (!runtimeURL.pathname.endsWith("/")) throw new Error("runtime_index_url_requires_trailing_slash");
  progress(requestId, "runtime", "首次打开需要下载 Python 计算环境；下载完成后可在浏览器重新计算。");
  const { loadPyodide } = await import(new URL("pyodide.mjs", runtimeURL).href);
  pyodide = await loadPyodide({ indexURL: runtimeURL.href });
  if (pyodide.version !== PYODIDE_VERSION) throw new Error("loaded_python_runtime_version_mismatch");
  progress(requestId, "numpy", "正在加载 NumPy 数值计算模块…");
  await pyodide.loadPackage("numpy");
  progress(requestId, "model", "正在校验训练模型权重与原仿真源码…");
  const [moduleBytes, modelBytes, bootstrapBytes] = await Promise.all([
    Promise.all(manifest.modules.map(record => verifiedResource(record))),
    verifiedResource(manifest.model),
    verifiedResource(manifest.bootstrap),
  ]);
  const pythonRoot = "/home/pyodide/browser_python";
  pyodide.FS.mkdirTree(`${pythonRoot}/delivery_risk`);
  manifest.modules.forEach((record, index) => {
    pyodide.FS.writeFile(`${pythonRoot}/${record.name}`, moduleBytes[index]);
  });
  const modelPath = "/home/pyodide/frozen_model.json";
  pyodide.FS.writeFile(modelPath, modelBytes);
  pyodide.globals.set("_browser_python_root", pythonRoot);
  pyodide.runPython("import sys\nsys.path.insert(0, _browser_python_root)");
  pyodide.runPython(new TextDecoder().decode(bootstrapBytes));
  const initializeModel = pyodide.globals.get("initialize_browser_model");
  try {
    info = JSON.parse(initializeModel(modelPath));
  } finally {
    initializeModel.destroy();
  }
  info.browser_runtime.pyodide = PYODIDE_VERSION;
  info.browser_runtime.compute_manifest_sha256 = manifestSha256;
  info.browser_runtime.model_json_sha256 = manifest.model.sha256;
  info.browser_runtime.model_download = manifest.model.url;
  return info;
}

async function ensureInitialized(requestId) {
  if (!initialization) {
    initialization = initialize(requestId).catch(error => {
      initialization = null;
      throw error;
    });
  }
  return await initialization;
}

self.onmessage = async event => {
  const message = event.data || {};
  const requestId = message.requestId;
  if (!["init", "run"].includes(message.type)) {
    self.postMessage({ type: "error", requestId, message: "unsupported_worker_request" });
    return;
  }
  if (message.type === "run" && busy) {
    self.postMessage({ type: "error", requestId, message: "simulation_busy_cancel_or_wait" });
    return;
  }
  const startedAt = performance.now();
  if (message.type === "run") busy = true;
  try {
    const initializedInfo = await ensureInitialized(requestId);
    if (message.type === "init") {
      self.postMessage({ type: "ready", requestId, info: initializedInfo });
      return;
    }
    if (!message.config || typeof message.config !== "object" || Array.isArray(message.config)) {
      throw new Error("simulation_config_object_required");
    }
    const configurationJSON = JSON.stringify(message.config);
    if (configurationJSON.length > 4096) throw new Error("simulation_config_too_large");
    const compare = message.compare === true;
    pyodide.globals.set("_browser_progress_callback", (phase, text, detailsJSON = "{}") => {
      progress(requestId, phase, text, { ...JSON.parse(detailsJSON), elapsedMs: performance.now() - startedAt });
    });
    const run = pyodide.globals.get("run_browser_simulation");
    let result;
    try {
      result = JSON.parse(run(configurationJSON, compare));
    } finally {
      run.destroy();
      pyodide.globals.delete("_browser_progress_callback");
    }
    const elapsedMs = performance.now() - startedAt;
    result.metadata.browser_compute = {
      fresh_simulation: true,
      saved_trajectory_used: false,
      engine: "pyodide_python_numpy_frozen_histgb_json",
      model_json_sha256: manifest.model.sha256,
      compute_manifest_sha256: manifestSha256,
      runtime_version: PYODIDE_VERSION,
      numpy_version: initializedInfo.browser_runtime.numpy,
      elapsed_ms: Math.round(elapsedMs),
      comparison_requested: compare,
    };
    self.postMessage({ type: "result", requestId, config: result.config, result, elapsedMs });
  } catch (error) {
    self.postMessage({ type: "error", requestId,
      message: String(error?.message || error),
      detail: String(error?.stack || "").slice(0, 8000) });
  } finally {
    if (message.type === "run") busy = false;
  }
};
