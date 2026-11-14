// Thin fetch wrapper for the FastAPI backend. No build step, no
// abstractions beyond "make the HTTP call and return parsed JSON."
const Api = {
  async get(path) {
    const r = await fetch(`/api${path}`);
    if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(`/api${path}`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      let detail = "";
      try {
        const err = await r.json();
        detail = err.detail ? (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)) : "";
      } catch (_) { /* ignore */ }
      throw new Error(`POST ${path} -> ${r.status}${detail ? `: ${detail}` : ""}`);
    }
    return r.json();
  },
  async patch(path, body) {
    const r = await fetch(`/api${path}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`PATCH ${path} -> ${r.status}`);
    return r.json();
  },
  async uploadCsv(name, file) {
    const form = new FormData();
    form.append("file", file);
    const r = await fetch(`/api/runs/datasets/upload?name=${encodeURIComponent(name)}`, {
      method: "POST", body: form,
    });
    if (!r.ok) throw new Error(`upload -> ${r.status}`);
    return r.json();
  },
};
