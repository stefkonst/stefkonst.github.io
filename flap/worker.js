// Runs model.py unchanged in the browser (Pyodide). Same code as the validation scripts.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");
let py = null;

async function init() {
  py = await loadPyodide();
  await py.loadPackage("numpy");
  const src = await (await fetch("model.py", { cache: "no-cache" })).text();
  py.FS.writeFile("model.py", src);
  py.runPython(`
import sys, json, js
sys.path.insert(0, ".")
import model
def _progress(done, total):
    js.postMessage(js.JSON.parse(json.dumps({"type": "progress", "done": done, "total": total})))
`);
  postMessage({ type: "ready", defaults: JSON.parse(py.runPython("json.dumps(model.DEFAULTS)")) });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (!py) await ready;
    py.globals.set("_args", JSON.stringify(m));
    if (m.type === "run") {
      const t = performance.now();
      const out = py.runPython("json.dumps(model.run(json.loads(_args)['params']), default=float)");
      postMessage({ type: "result", data: JSON.parse(out), ms: performance.now() - t });
    } else if (m.type === "sweep") {
      const out = py.runPython(`
_a = json.loads(_args)
json.dumps(model.sweep(_a['params'], _a['kind'], _a['values'], voltages=_a['voltages'], aoas=_a['aoas'], progress=_progress), default=float)`);
      postMessage({ type: "sweep", data: JSON.parse(out) });
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err) });
  }
};
const ready = init().catch((err) => postMessage({ type: "error", message: "Could not start Python: " + err }));
